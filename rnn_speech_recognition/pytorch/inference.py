# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import os
import random
import time
import tqdm
from heapq import nlargest
from itertools import chain, repeat
from pathlib import Path

import dllogger
import torch
import numpy as np
import torch.distributed as distrib
from apex import amp
from apex.parallel import DistributedDataParallel
from dllogger import JSONStreamBackend, StdOutBackend, Verbosity

from common import helpers
from common.data import features
from common.data.dali import sampler as dali_sampler
from common.data.dali.data_loader import DaliDataLoader
from common.data.features import BaseFeatures, FilterbankFeatures
from common.data.text import Tokenizer
from common.helpers import print_once, process_evaluation_epoch
from common.tb_dllogger import stdout_metric_format, unique_log_fpath
from rnnt import config
from rnnt.decoder import RNNTGreedyDecoder
from rnnt.model import RNNT


def get_parser():
    parser = argparse.ArgumentParser(description='RNN-T')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Data batch size')
    parser.add_argument('--steps', default=0, type=int,
                        help='Eval this many steps for every worker')
    parser.add_argument('--model_config', type=str,
                        help='Relative model config path given dataset folder')
    parser.add_argument('--dataset_dir', type=str,
                        help='Absolute path to dataset folder')
    parser.add_argument('--val_manifest', type=str,
                        help='Relative path to evaluation dataset manifest file')
    parser.add_argument('--ckpt', default=None, type=str,
                        help='Path to model checkpoint')
    parser.add_argument('--max_duration', default=None, type=float,
                        help='Filter out longer inputs (in seconds)')
    parser.add_argument('--pad_to_max_duration', action='store_true',
                        help='Pads every batch to max_duration')
    parser.add_argument('--amp', '--fp16', action='store_true',
                        help='Use FP16 precision')
    parser.add_argument('--cudnn_benchmark', action='store_true',
                        help='Enable cudnn benchmark')
    parser.add_argument('--save_predictions', type=str, default=None,
                        help='Save predictions in text form at this location')
    parser.add_argument('--transcribe_wav', type=str,
                        help='Path to a single .wav file (16KHz)')
    parser.add_argument('--transcribe_filelist', type=str,
                        help='Path to a filelist with one .wav path per line')
    parser.add_argument('--dali_device', type=str, choices=['none', 'cpu', 'gpu'],
                        default='gpu', help='')  # XXX
    parser.add_argument('--repeats', default=1, type=int,
                        help='Repeat the inference for benchmarking')

    parser.add_argument('-o', '--output_dir', default='results/',
                        help='Output folder to save audio (file per phrase)')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to a DLLogger log file')
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0),
                        type=int, help='GPU id used for distributed training')

    parser.add_argument('--cpu', action='store_true',
                        help='Run inference on CPU')
    parser.add_argument('--ema', action='store_true',
                        help='Load EMA model weights')

    parser.add_argument("--seed", default=None, type=int, help='seed')

    return parser


def durs_to_percentiles(durations, ratios):
    durations = np.asarray(durations) * 1000  # in ms
    latency = durations

    latency = latency[5:]
    mean_latency = np.mean(latency)

    latency_worst = nlargest(math.ceil((1 - min(ratios))* len(latency)), latency)
    latency_ranges = get_percentile(ratios, latency_worst, len(latency))
    latency_ranges[0.5] = mean_latency
    return latency_ranges


def get_percentile(ratios, arr, nsamples):
    res = {}
    for a in ratios:
        idx = max(int(nsamples * (1 - a)), 0)
        res[a] = arr[idx]
    return res


def main():
    parser = get_parser()
    args = parser.parse_args()

    log_fpath = args.log_file or str(Path(args.output_dir, 'nvlog_infer.json'))
    log_fpath = unique_log_fpath(log_fpath)
    dllogger.init(backends=[JSONStreamBackend(Verbosity.DEFAULT, log_fpath),
                            StdOutBackend(Verbosity.VERBOSE,
                                          metric_format=stdout_metric_format)])

    [dllogger.log("PARAMETER", {k:v}) for k,v in vars(args).items()]

    for step in ['DNN', 'data+DNN', 'data']:
        for c in [0.99, 0.95, 0.9, 0.5]:
            cs = 'avg' if c == 0.5 else f'{int(100*c)}%'
            dllogger.metadata(f'{step.lower()}_latency_{c}',
                              {'name': f'{step} latency {cs}',
                               'format': ':>7.2f', 'unit': 'ms'})
    dllogger.metadata(
        'eval_wer', {'name': 'WER', 'format': ':>3.3f', 'unit': '%'})

    if args.cpu:
        device = torch.device('cpu')
    else:
        assert torch.cuda.is_available()
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = args.cudnn_benchmark

    if args.seed is not None:
        torch.manual_seed(args.seed + args.local_rank)
        np.random.seed(args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

    # set up distributed training
    multi_gpu = not args.cpu and int(os.environ.get('WORLD_SIZE', 1)) > 1
    if multi_gpu:
        torch.cuda.set_device(args.local_rank)
        distrib.init_process_group(backend='nccl', init_method='env://')
        print_once(f'Inference with {distrib.get_world_size()} GPUs')

    cfg = config.load(args.model_config)

    if args.max_duration is not None:
        cfg['input_val']['audio_dataset']['max_duration'] = args.max_duration
        cfg['input_val']['filterbank_features']['max_duration'] = args.max_duration

    if args.pad_to_max_duration:
        assert cfg['input_val']['audio_dataset']['max_duration'] > 0
        cfg['input_val']['audio_dataset']['pad_to_max_duration'] = True
        cfg['input_val']['filterbank_features']['pad_to_max_duration'] = True

    use_dali = args.dali_device in ('cpu', 'gpu')

    (
        dataset_kw,
        features_kw,
        splicing_kw,
        _, _
    ) = config.input(cfg, 'val')

    tokenizer_kw = config.tokenizer(cfg)
    tokenizer = Tokenizer(**tokenizer_kw)

    optim_level = 3 if args.amp else 0

    feature_proc  = torch.nn.Sequential(
        torch.nn.Identity(),
        torch.nn.Identity(),
        features.FrameSplicing(optim_level=optim_level, **splicing_kw),
        features.FillPadding(optim_level=optim_level, ),
    )

    # dataset

    data_loader = DaliDataLoader(
        gpu_id=args.local_rank or 0,
        dataset_path=args.dataset_dir,
        config_data=dataset_kw,
        config_features=features_kw,
        json_names=[args.val_manifest],
        batch_size=args.batch_size,
        sampler=dali_sampler.SimpleSampler(),
        pipeline_type="val",
        device_type=args.dali_device,
        tokenizer=tokenizer)

    model = RNNT(n_classes=tokenizer.num_labels + 1, **config.rnnt(cfg))

    if args.ckpt is not None:
        print(f'Loading the model from {args.ckpt} ...')
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        key = 'ema_state_dict' if args.ema else 'state_dict'
        state_dict = checkpoint[key]
        model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    if feature_proc is not None:
        feature_proc.to(device)
        feature_proc.eval()

    if args.amp:
        model = amp.initialize(model, opt_level='O3')

    if multi_gpu:
        model = DistributedDataParallel(model)

    agg = {'txts': [], 'preds': [], 'logits': []}
    dur = {'data': [], 'dnn': [], 'data+dnn': []}

    rep_loader = chain(*repeat(data_loader, args.repeats))
    rep_len = args.repeats * len(data_loader)

    blank_idx = tokenizer.num_labels
    greedy_decoder = RNNTGreedyDecoder(blank_idx=blank_idx)

    def sync_time():
        torch.cuda.synchronize() if device.type == 'cuda' else None
        return time.perf_counter()

    sz = []
    with torch.no_grad():

        for it, batch in enumerate(tqdm.tqdm(rep_loader, total=rep_len)):

            if use_dali:
                feats, feat_lens, txt, txt_lens = batch
                if feature_proc is not None:
                    feats, feat_lens = feature_proc([feats, feat_lens])
            else:
                batch = [t.cuda(non_blocking=True) for t in batch]
                audio, audio_lens, txt, txt_lens = batch
                feats, feat_lens = feature_proc([audio, audio_lens])
            feats = feats.permute(2, 0, 1)
            if args.amp:
                feats = feats.half()

            sz.append(feats.size(0))

            t1 = sync_time()
            log_probs, log_prob_lens = model(feats, feat_lens, txt, txt_lens)
            t2 = sync_time()

            # burn-in period; wait for a new loader due to num_workers
            if it >= 1 and (args.repeats == 1 or it >= len(data_loader)):
                dur['data'].append(t1 - t0)
                dur['dnn'].append(t2 - t1)
                dur['data+dnn'].append(t2 - t0)

            if txt is not None:
                agg['txts'] += helpers.gather_transcripts([txt], [txt_lens],
                                                          tokenizer.detokenize)

            preds = greedy_decoder.decode(model, feats, feat_lens)

            agg['preds'] += helpers.gather_predictions([preds], tokenizer.detokenize)

            if 0 < args.steps < it:
                break

            t0 = sync_time()

        # communicate the results
        if args.transcribe_wav:
            for idx,p in enumerate(agg['preds']):
                print_once(f'Prediction {idx+1: >3}: {p}')

        elif args.transcribe_filelist:
            pass

        else:
            wer, loss = process_evaluation_epoch(agg)

            if not multi_gpu or distrib.get_rank() == 0:
                dllogger.log(step=(), data={'eval_wer': 100 * wer})

        if args.save_predictions:
            with open(args.save_predictions, 'w') as f:
                f.write('\n'.join(agg['preds']))

    # report timings
    if len(dur['data']) >= 20:
        ratios = [0.9, 0.95, 0.99]

        for stage in dur:
            lat = durs_to_percentiles(dur[stage], ratios)
            for k in [0.99, 0.95, 0.9, 0.5]:
                kk = str(k).replace('.', '_')
                dllogger.log(step=(), data={f'{stage.lower()}_latency_{kk}': lat[k]})

    else:
        # TODO measure at least avg latency
        print_once('Not enough samples to measure latencies.')


if __name__=="__main__":
    main()
