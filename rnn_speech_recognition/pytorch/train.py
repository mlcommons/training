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
import copy
import os
import random
import time

import torch
import numpy as np
import torch.distributed as dist
from apex import amp
from apex.optimizers import FusedAdam, FusedLAMB
from apex.parallel import DistributedDataParallel

from common import helpers
from common.data.dali import sampler as dali_sampler
from common.data.dali.data_loader import DaliDataLoader
from common.data.dataset import AudioDataset, get_data_loader
from common.data.text import Tokenizer
from common.data import features
from common.data import dataset_size
from common.helpers import (Checkpointer, greedy_wer, num_weights, print_once,
                            process_evaluation_epoch)
from common.optimizers import AdamW, lr_policy, Novograd
from common.tb_dllogger import flush_log, init_log, log
from rnnt import config
from rnnt.decoder import RNNTGreedyDecoder
from rnnt.loss import RNNTLoss
from rnnt.model import RNNT

from mlperf import logging


# TODO Eval batch size

def parse_args():
    parser = argparse.ArgumentParser(description='RNN-T Training Reference')

    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', default=400, type=int,
                          help='Number of epochs for the entire training; influences the lr schedule')
    training.add_argument("--warmup_epochs", default=0, type=int,
                          help='Initial epochs of increasing learning rate')
    training.add_argument("--hold_epochs", default=0, type=int,
                          help='Constant max learning rate epochs after warmup')
    training.add_argument('--epochs_this_job', default=0, type=int,
                          help=('Run for a number of epochs with no effect on the lr schedule.'
                                'Useful for re-starting the training.'))
    training.add_argument('--cudnn_benchmark', action='store_true', default=True,
                          help='Enable cudnn benchmark')
    training.add_argument('--amp', '--fp16', action='store_true', default=False,
                          help='Use mixed precision training')
    training.add_argument('--seed', default=42, type=int, help='Random seed')
    training.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int,
                          help='GPU id used for distributed training')

    optim = parser.add_argument_group('optimization setup')
    optim.add_argument('--batch_size', default=32, type=int,
                       help='Global batch size')
    optim.add_argument('--val_batch_size', default=2, type=int,
                       help='Evalution time batch size')
    optim.add_argument('--lr', default=1e-3, type=float,
                       help='Peak learning rate')
    optim.add_argument("--min_lr", default=1e-5, type=float,
                       help='minimum learning rate')
    optim.add_argument("--lr_policy", default='exponential', type=str,
                       choices=['exponential', 'legacy', 'transformer'], help='lr scheduler')
    optim.add_argument("--lr_exp_gamma", default=0.99, type=float,
                       help='gamma factor for exponential lr scheduler')
    optim.add_argument('--weight_decay', default=1e-3, type=float,
                       help='Weight decay for the optimizer')
    optim.add_argument('--grad_accumulation_steps', default=1, type=int,
                       help='Number of accumulation steps')
    optim.add_argument('--optimizer', default='novograd', type=str,
                       choices=['novograd', 'adamw', 'adam', 'lamb', 'adam98', 'lamb98'],
                       help='Optimization algorithm')
    optim.add_argument('--ema', type=float, default=0.0,
                       help='Discount factor for exp averaging of model weights')

    io = parser.add_argument_group('feature and checkpointing setup')
    io.add_argument('--dali_device', type=str, choices=['none', 'cpu', 'gpu'],
                    default='gpu', help='Use DALI pipeline for fast data processing')
    io.add_argument('--resume', action='store_true',
                    help='Try to resume from last saved checkpoint.')
    io.add_argument('--ckpt', default=None, type=str,
                    help='Path to a checkpoint for resuming training')
    io.add_argument('--save_frequency', default=10, type=int,
                    help='Checkpoint saving frequency in epochs')
    io.add_argument('--keep_milestones', default=[100, 200, 300], type=int, nargs='+',
                    help='Milestone checkpoints to keep from removing')
    io.add_argument('--save_best_from', default=380, type=int,
                    help='Epoch on which to begin tracking best checkpoint (dev WER)')
    io.add_argument('--val_frequency', default=200, type=int,
                    help='Number of steps between evaluations on dev set')
    io.add_argument('--log_frequency', default=25, type=int,
                    help='Number of steps between printing training stats')
    io.add_argument('--prediction_frequency', default=100, type=int,
                    help='Number of steps between printing sample decodings')
    io.add_argument('--model_config', type=str, required=True,
                    help='Path of the model configuration file')
    io.add_argument('--train_manifests', type=str, required=True, nargs='+',
                    help='Paths of the training dataset manifest file')
    io.add_argument('--val_manifests', type=str, required=True, nargs='+',
                    help='Paths of the evaluation datasets manifest files')
    io.add_argument('--max_duration', type=float,
                    help='Discard samples longer than max_duration')
    io.add_argument('--legacy_bucketing', action='store_true',
                    help='buckets are clipped to multiple of batch_size')
    io.add_argument('--num_buckets', type=int, default=None,
                    help='If provided, samples will be grouped by audio duration, '
                         'to this number of backets, for each bucket, '
                         'random samples are batched, and finally '
                         'all batches are randomly shuffled')
    io.add_argument('--dataset_dir', required=True, type=str,
                    help='Root dir of dataset')
    io.add_argument('--output_dir', type=str, required=True,
                    help='Directory for logs and checkpoints')
    io.add_argument('--log_file', type=str, default=None,
                    help='Path to save the training logfile.')
    return parser.parse_args()


def barrier():
    """
    Works as a temporary distributed barrier, currently pytorch
    doesn't implement barrier for NCCL backend.
    Calls all_reduce on dummy tensor and synchronizes with GPU.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(torch.cuda.FloatTensor(1))
        torch.cuda.synchronize()


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt.true_divide(num_gpus)


def apply_ema(model, ema_model, decay):
    if not decay:
        return

    sd = getattr(model, 'module', model).state_dict()
    for k, v in ema_model.state_dict().items():
        v.copy_(decay * v + (1 - decay) * sd[k])


@torch.no_grad()
def evaluate(epoch, step, val_loader, val_feat_proc, detokenize, model,
             ema_model, loss_fn, greedy_decoder, use_amp, use_dali=False):

    for model, subset in [(ema_model, 'dev_ema')]:
        if model is None:
            continue

        logging.log_start(logging.constants.EVAL_START,
                          metadata=dict(epoch_num=epoch))

        model.eval()
        start_time = time.time()
        agg = {'losses': [], 'preds': [], 'txts': []}

        for i, batch in enumerate(val_loader):
            print(f'evaluation: {i:>10}/{len(val_loader):<10}', end='\r')

            if not use_dali:
                batch = [t.cuda(non_blocking=True) for t in batch]
            audio, audio_lens, txt, txt_lens = batch

            feats, feat_lens = val_feat_proc([audio, audio_lens])

            log_probs, log_prob_lens = model(feats, feat_lens, txt, txt_lens)
            loss = loss_fn(log_probs[:, :log_prob_lens.max().item()],
                                      log_prob_lens, txt, txt_lens)

            pred = greedy_decoder.decode(model, feats, feat_lens)

            agg['losses'] += helpers.gather_losses([loss.cpu()])
            agg['preds'] += helpers.gather_predictions([pred], detokenize)
            agg['txts'] += helpers.gather_transcripts([txt.cpu()], [txt_lens.cpu()], detokenize)

        wer, loss = process_evaluation_epoch(agg)

        logging.log_end(logging.constants.EVAL_STOP)

        logging.log_event(logging.constants.EVAL_ACCURACY,
                          value=wer,
                          metadata=dict(epoch_num=epoch))
        logging.log_end(logging.constants.EVAL_STOP,
                        metadata=dict(epoch_num=epoch))

        log((epoch,), step, subset, {'loss': loss, 'wer': 100.0 * wer,
                                     'took': time.time() - start_time})
        model.train()
    return wer


def main():
    logging.configure_logger('RNNT')
    logging.log_start(logging.constants.INIT_START)

    args = parse_args()

    assert(torch.cuda.is_available())
    assert args.prediction_frequency % args.log_frequency == 0

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    # set up distributed training
    multi_gpu = int(os.environ.get('WORLD_SIZE', 1)) > 1
    if multi_gpu:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        world_size = dist.get_world_size()
        print_once(f'Distributed training with {world_size} GPUs\n')
    else:
        world_size = 1

    logging.log_event(logging.constants.SEED, value=args.seed)
    torch.manual_seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    random.seed(args.seed + args.local_rank)
    np_rng = np.random.default_rng(seed=args.seed + args.local_rank)

    init_log(args)

    cfg = config.load(args.model_config)
    config.apply_duration_flags(cfg, args.max_duration)

    assert args.grad_accumulation_steps >= 1
    assert args.batch_size % args.grad_accumulation_steps == 0
    batch_size = args.batch_size // args.grad_accumulation_steps

    logging.log_end(logging.constants.INIT_STOP)
    barrier()
    logging.log_start(logging.constants.RUN_START)
    barrier()

    logging.log_event(logging.constants.MAX_SEQUENCE_LENGTH, value=args.max_duration)
    logging.log_event(logging.constants.GLOBAL_BATCH_SIZE,
                      value=batch_size * world_size)

    print_once('Setting up datasets...')
    (
        train_dataset_kw,
        train_features_kw,
        train_splicing_kw,
        train_specaugm_kw,
        train_cutoutau_kw,
    ) = config.input(cfg, 'train')
    (
        val_dataset_kw,
        val_features_kw,
        val_splicing_kw,
        val_specaugm_kw,
        val_cutoutau_kw,
    ) = config.input(cfg, 'val')

    tokenizer_kw = config.tokenizer(cfg)
    tokenizer = Tokenizer(**tokenizer_kw)

    class PermuteAudio(torch.nn.Module):
        def forward(self, x):
            return (x[0].permute(2, 0, 1), *x[1:])

    train_augmentations = torch.nn.Sequential(
        train_cutoutau_kw and features.CutoutAugment(optim_level=args.amp, **train_cutoutau_kw) or torch.nn.Identity(),
        train_specaugm_kw and features.SpecAugment(optim_level=args.amp, **train_specaugm_kw) or torch.nn.Identity(),
        features.FrameSplicing(optim_level=args.amp, **train_splicing_kw),
        features.FillPadding(optim_level=args.amp, ),
        PermuteAudio(),
    )
    val_augmentations = torch.nn.Sequential(
        val_cutoutau_kw and features.CutoutAugment(optim_level=args.amp, **val_cutoutau_kw) or torch.nn.Identity(),
        val_specaugm_kw and features.SpecAugment(optim_level=args.amp, **val_specaugm_kw) or torch.nn.Identity(),
        features.FrameSplicing(optim_level=args.amp, **val_splicing_kw),
        features.FillPadding(optim_level=args.amp, ),
        PermuteAudio(),
    )

    use_dali = args.dali_device in ('cpu', 'gpu')
    if use_dali:

        if args.num_buckets is not None:
            sampler = dali_sampler.BucketingSampler(
                args.num_buckets,
                batch_size*world_size,
                args.epochs,
                np_rng
            )
        else:
            sampler = dali_sampler.SimpleSampler()

        train_loader = DaliDataLoader(gpu_id=args.local_rank,
                                      dataset_path=args.dataset_dir,
                                      config_data=train_dataset_kw,
                                      config_features=train_features_kw,
                                      json_names=args.train_manifests,
                                      batch_size=batch_size,
                                      sampler=sampler,
                                      grad_accumulation_steps=args.grad_accumulation_steps,
                                      pipeline_type="train",
                                      device_type=args.dali_device,
                                      tokenizer=tokenizer)

        val_loader = DaliDataLoader(gpu_id=args.local_rank,
                                    dataset_path=args.dataset_dir,
                                    config_data=val_dataset_kw,
                                    config_features=val_features_kw,
                                    json_names=args.val_manifests,
                                    batch_size=args.val_batch_size,
                                    sampler=dali_sampler.SimpleSampler(),
                                    pipeline_type="val",
                                    device_type=args.dali_device,
                                    tokenizer=tokenizer)

        train_feat_proc = train_augmentations
        val_feat_proc   = val_augmentations
    else:
        train_dataset = AudioDataset(args.dataset_dir,
                                     args.train_manifests,
                                     tokenizer=tokenizer,
                                     **train_dataset_kw)
        train_loader = get_data_loader(train_dataset,
                                       batch_size,
                                       world_size,
                                       args.local_rank,
                                       num_buckets=args.num_buckets,
                                       shuffle=True,
                                       num_workers=4)

        val_dataset = AudioDataset(args.dataset_dir,
                                   args.val_manifests,
                                   tokenizer=tokenizer,
                                   **val_dataset_kw)
        val_loader = get_data_loader(val_dataset,
                                     args.val_batch_size,
                                     world_size,
                                     args.local_rank,
                                     shuffle=False,
                                     num_workers=4,
                                     drop_last=False)

        train_feat_proc = torch.nn.Sequential(
            features.FilterbankFeatures(optim_level=args.amp, **train_features_kw),
            train_augmentations,
        )
        val_feat_proc = torch.nn.Sequential(
            features.FilterbankFeatures(optim_level=args.amp, **val_features_kw),
            val_augmentations,
        )

        dur = train_dataset.duration / 3600
        dur_f = train_dataset.duration_filtered / 3600
        nsampl = len(train_dataset)
        print_once(f'Training samples: {nsampl} ({dur:.1f}h, '
                   f'filtered {dur_f:.1f}h)')

    train_feat_proc.cuda()
    val_feat_proc.cuda()

    steps_per_epoch = len(train_loader) // args.grad_accumulation_steps

    logging.log_event(logging.constants.TRAIN_SAMPLES, value=dataset_size(train_loader))
    logging.log_event(logging.constants.EVAL_SAMPLES, value=len(val_loader))

    # set up the model
    model = RNNT(n_classes=tokenizer.num_labels + 1, **config.rnnt(cfg))
    model.cuda()
    blank_idx = tokenizer.num_labels
    loss_fn = RNNTLoss(blank_idx=blank_idx)
    greedy_decoder = RNNTGreedyDecoder(blank_idx=blank_idx)

    print_once(f'Model size: {num_weights(model) / 10**6:.1f}M params\n')

    logging.log_event(logging.constants.OPT_NAME, value=args.optimizer)
    logging.log_event(logging.constants.OPT_BASE_LR, value=args.lr)

    # optimization
    kw = {'params': model.param_groups(args.lr), 'lr': args.lr,
          'weight_decay': args.weight_decay}

    initial_lrs = [group['lr'] for group in kw['params']]

    print_once(f'Starting with LRs: {initial_lrs}')

    if args.optimizer == "novograd":
        optimizer = Novograd(**kw)
    elif args.optimizer == "adamw":
        optimizer = AdamW(**kw)
    elif args.optimizer == 'adam':
        optimizer = FusedAdam(**kw)
    elif args.optimizer == 'lamb':
        optimizer = FusedLAMB(**kw)
    elif args.optimizer == 'adam98':
        optimizer = FusedAdam(betas=(0.9, 0.98), eps=1e-9, **kw)
    elif args.optimizer == 'lamb98':
        optimizer = FusedLAMB(betas=(0.9, 0.98), eps=1e-9, **kw)
    else:
        raise ValueError(f'Invalid optimizer "{args.optimizer}"')

    adjust_lr = lambda step, epoch: lr_policy(
        step, epoch, initial_lrs, optimizer, steps_per_epoch=steps_per_epoch,
        warmup_epochs=args.warmup_epochs, hold_epochs=args.hold_epochs,
        num_epochs=args.epochs, policy=args.lr_policy, min_lr=args.min_lr,
        exp_gamma=args.lr_exp_gamma)

    if args.amp:
        model, optimizer = amp.initialize(
            models=model,
            optimizers=optimizer,
            opt_level='O1',
            max_loss_scale=512.0)

    if args.ema > 0:
        ema_model = copy.deepcopy(model).cuda()
    else:
        ema_model = None

    if multi_gpu:
        model = DistributedDataParallel(model)

    # load checkpoint
    meta = {'best_wer': 10**6, 'start_epoch': 0}
    checkpointer = Checkpointer(args.output_dir, 'RNN-T',
                                args.keep_milestones, args.amp)
    if args.resume:
        args.ckpt = checkpointer.last_checkpoint() or args.ckpt

    if args.ckpt is not None:
        checkpointer.load(args.ckpt, model, ema_model, optimizer, meta)

    start_epoch = meta['start_epoch']
    best_wer = meta['best_wer']
    epoch = 1
    step = start_epoch * steps_per_epoch + 1

    # training loop
    model.train()
    for epoch in range(start_epoch + 1, args.epochs + 1):

        logging.log_start(logging.constants.BLOCK_START,
                          metadata=dict(first_epoch_num=epoch,
                                        epoch_count=1))
        logging.log_start(logging.constants.EPOCH_START,
                          metadata=dict(epoch_num=epoch))

        if multi_gpu and not use_dali:
            train_loader.sampler.set_epoch(epoch)

        epoch_utts = 0
        accumulated_batches = 0
        epoch_start_time = time.time()

        for batch in train_loader:

            if accumulated_batches == 0:
                adjust_lr(step, epoch)
                optimizer.zero_grad()
                step_loss = 0
                step_utts = 0
                step_start_time = time.time()

            if not use_dali:
                batch = [t.cuda(non_blocking=True) for t in batch]
            audio, audio_lens, txt, txt_lens = batch

            feats, feat_lens = train_feat_proc([audio, audio_lens])

            log_probs, log_prob_lens = model(feats, feat_lens, txt, txt_lens)
            loss = loss_fn(log_probs[:, :log_prob_lens.max().item()],
                                      log_prob_lens, txt, txt_lens)

            loss /= args.grad_accumulation_steps

            # TODO why ? is decoding memory-intensive?
            del log_probs

            if torch.isnan(loss).any():
                print_once(f'WARNING: loss is NaN; skipping update')
            else:
                if multi_gpu:
                    step_loss += reduce_tensor(loss.data, world_size).item()
                else:
                    step_loss += loss.item()

                if args.amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                step_utts += batch[0].size(0) * world_size
                epoch_utts += batch[0].size(0) * world_size
                accumulated_batches += 1
                assert args.grad_accumulation_steps == 1, "Probes do not yet support accumulation"

            if accumulated_batches % args.grad_accumulation_steps == 0:
                optimizer.step()
                apply_ema(model, ema_model, args.ema)

                if step % args.log_frequency == 0:

                    if step % args.prediction_frequency == 0:
                        preds = greedy_decoder.decode(model, feats, feat_lens)
                        wer, pred_utt, ref = greedy_wer(
                                preds,
                                txt,
                                txt_lens,
                                tokenizer.detokenize)
                        print_once(f'  Decoded:   {pred_utt[:90]}')
                        print_once(f'  Reference: {ref[:90]}')
                        wer = {'wer': 100 * wer}
                    else:
                        wer = {}

                    step_time = time.time() - step_start_time
                    log((epoch, step % steps_per_epoch or steps_per_epoch, steps_per_epoch),
                        step, 'train',
                        {'loss': loss.item(),
                         **wer,  # optional entry
                         'throughput': step_utts / step_time,
                         'took': step_time,
                         'lrate': optimizer.param_groups[0]['lr']})

                step_start_time = time.time()

                if step % args.val_frequency == 0:
                    wer = evaluate(epoch, step, val_loader, val_feat_proc,
                                   tokenizer.detokenize, model, ema_model, loss_fn,
                                   greedy_decoder, args.amp, use_dali)

                    if wer < best_wer and epoch >= args.save_best_from:
                        checkpointer.save(model, ema_model, optimizer, epoch,
                                          step, best_wer, is_best=True)
                        best_wer = wer

                step += 1
                accumulated_batches = 0
                # end of step

            # DALI iterator need to be exhausted;
            # if not using DALI, simulate drop_last=True with grad accumulation
            if not use_dali and step > steps_per_epoch * epoch:
                break

        logging.log_end(logging.constants.EPOCH_STOP,
                        metadata=dict(epoch_num=epoch))

        logging.log_end(logging.constants.BLOCK_STOP, metadata=dict(first_epoch_num=epoch))

        epoch_time = time.time() - epoch_start_time
        log((epoch,), None, 'train_avg', {'throughput': epoch_utts / epoch_time,
                                          'took': epoch_time})

        if epoch % args.save_frequency == 0 or epoch in args.keep_milestones:
            checkpointer.save(model, ema_model, optimizer, epoch, step, best_wer)

        if 0 < args.epochs_this_job <= epoch - start_epoch:
            print_once(f'Finished after {args.epochs_this_job} epochs.')
            break
        # end of epoch

    log((), None, 'train_avg', {'throughput': epoch_utts / epoch_time})

    if epoch == args.epochs:
        evaluate(epoch, step, val_loader, val_feat_proc, tokenizer.detokenize, model,
                 ema_model, loss_fn, greedy_decoder, args.amp, use_dali)

    flush_log()
    checkpointer.save(model, ema_model, optimizer, epoch, step, best_wer)


if __name__ == "__main__":
    main()
