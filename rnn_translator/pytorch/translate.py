#!/usr/bin/env python
import argparse
import codecs
import time
import warnings
from ast import literal_eval
from itertools import zip_longest

import torch

from seq2seq import models
from seq2seq.inference.inference import Translator
from seq2seq.utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description='GNMT Translate',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data
    dataset = parser.add_argument_group('data setup')
    dataset.add_argument('-i', '--input', required=True,
                         help='input file (tokenized)')
    dataset.add_argument('-o', '--output', required=True,
                         help='output file (tokenized)')
    dataset.add_argument('-m', '--model', required=True,
                         help='model checkpoint file')
    # parameters
    params = parser.add_argument_group('inference setup')
    params.add_argument('--batch-size', default=128, type=int,
                        help='batch size')
    params.add_argument('--beam-size', default=5, type=int,
                        help='beam size')
    params.add_argument('--max-seq-len', default=80, type=int,
                        help='maximum prediciton sequence length')
    params.add_argument('--len-norm-factor', default=0.6, type=float,
                        help='length normalization factor')
    params.add_argument('--cov-penalty-factor', default=0.1, type=float,
                        help='coverage penalty factor')
    params.add_argument('--len-norm-const', default=5.0, type=float,
                        help='length normalization constant')
    # general setup
    general = parser.add_argument_group('general setup')
    general.add_argument('--math', default='fp16', choices=['fp32', 'fp16'],
                         help='arithmetic type')

    batch_first_parser = general.add_mutually_exclusive_group(required=False)
    batch_first_parser.add_argument('--batch-first', dest='batch_first',
                                    action='store_true',
                                    help='uses (batch, seq, feature) data \
                                    format for RNNs')
    batch_first_parser.add_argument('--seq-first', dest='batch_first',
                                    action='store_false',
                                    help='uses (seq, batch, feature) data \
                                    format for RNNs')
    batch_first_parser.set_defaults(batch_first=True)

    cuda_parser = general.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true',
                             help='enables cuda (use \'--no-cuda\' to disable)')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                             help=argparse.SUPPRESS)
    cuda_parser.set_defaults(cuda=True)

    cudnn_parser = general.add_mutually_exclusive_group(required=False)
    cudnn_parser.add_argument('--cudnn', dest='cudnn', action='store_true',
                              help='enables cudnn (use \'--no-cudnn\' to disable)')
    cudnn_parser.add_argument('--no-cudnn', dest='cudnn', action='store_false',
                              help=argparse.SUPPRESS)
    cudnn_parser.set_defaults(cudnn=True)

    general.add_argument('--print-freq', '-p', default=1, type=int,
                         help='print log every PRINT_FREQ batches')

    return parser.parse_args()


def grouper(iterable, size, fillvalue=None):
    args = [iter(iterable)] * size
    return zip_longest(*args, fillvalue=fillvalue)


def write_output(output_file, lines):
    for line in lines:
        output_file.write(line)
        output_file.write('\n')


def checkpoint_from_distributed(state_dict):
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret


def unwrap_distributed(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value

    return new_state_dict


def main():
    args = parse_args()
    print(args)

    if args.cuda:
        torch.cuda.set_device(0)
    if not args.cuda and torch.cuda.is_available():
        warnings.warn('cuda is available but not enabled')
    if args.math == 'fp16' and not args.cuda:
        raise RuntimeError('fp16 requires cuda')
    if not args.cudnn:
        torch.backends.cudnn.enabled = False

    checkpoint = torch.load(args.model, map_location={'cuda:0': 'cpu'})

    vocab_size = checkpoint['tokenizer'].vocab_size
    model_config = dict(vocab_size=vocab_size, math=checkpoint['config'].math,
                        **literal_eval(checkpoint['config'].model_config))
    model_config['batch_first'] = args.batch_first
    model = models.GNMT(**model_config)

    state_dict = checkpoint['state_dict']
    if checkpoint_from_distributed(state_dict):
        state_dict = unwrap_distributed(state_dict)

    model.load_state_dict(state_dict)

    if args.math == 'fp32':
        dtype = torch.FloatTensor
    if args.math == 'fp16':
        dtype = torch.HalfTensor

    model.type(dtype)
    if args.cuda:
        model = model.cuda()
    model.eval()

    tokenizer = checkpoint['tokenizer']

    translation_model = Translator(model,
                                   tokenizer,
                                   beam_size=args.beam_size,
                                   max_seq_len=args.max_seq_len,
                                   len_norm_factor=args.len_norm_factor,
                                   len_norm_const=args.len_norm_const,
                                   cov_penalty_factor=args.cov_penalty_factor,
                                   cuda=args.cuda)

    output_file = codecs.open(args.output, 'w', encoding='UTF-8')

    # run model on generated data, for accurate timings starting from 1st batch
    dummy_data = ['abc ' * (args.max_seq_len // 4)] * args.batch_size
    translation_model.translate(dummy_data)

    if args.cuda:
        torch.cuda.synchronize()

    batch_time = AverageMeter(False)
    enc_tok_per_sec = AverageMeter(False)
    dec_tok_per_sec = AverageMeter(False)
    tot_tok_per_sec = AverageMeter(False)

    enc_seq_len = AverageMeter(False)
    dec_seq_len = AverageMeter(False)

    total_lines = 0
    total_iters = 0
    with codecs.open(args.input, encoding='UTF-8') as input_file:
        for idx, lines in enumerate(grouper(input_file, args.batch_size)):
            lines = [l for l in lines if l]
            n_lines = len(lines)
            total_lines += n_lines

            translate_timer = time.time()
            translated_lines, stats = translation_model.translate(lines)
            elapsed = time.time() - translate_timer

            batch_time.update(elapsed, n_lines)
            etps = stats['total_enc_len'] / elapsed
            dtps = stats['total_dec_len'] / elapsed
            enc_seq_len.update(stats['total_enc_len'] / n_lines, n_lines)
            dec_seq_len.update(stats['total_dec_len'] / n_lines, n_lines)
            enc_tok_per_sec.update(etps, n_lines)
            dec_tok_per_sec.update(dtps, n_lines)

            tot_tok = stats['total_dec_len'] + stats['total_enc_len']
            ttps = tot_tok / elapsed
            tot_tok_per_sec.update(ttps, n_lines)

            n_iterations = stats['iters']
            total_iters += n_iterations

            write_output(output_file, translated_lines)

            if idx % args.print_freq == args.print_freq - 1:
                print(f'TRANSLATION: '
                      f'Batch {idx} '
                      f'Iters {n_iterations}\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Tot tok/s {tot_tok_per_sec.val:.0f} ({tot_tok_per_sec.avg:.0f})\t'
                      f'Enc tok/s {enc_tok_per_sec.val:.0f} ({enc_tok_per_sec.avg:.0f})\t'
                      f'Dec tok/s {dec_tok_per_sec.val:.0f} ({dec_tok_per_sec.avg:.0f})')

    output_file.close()

    print(f'TRANSLATION SUMMARY:\n'
          f'Lines translated: {total_lines}\t'
          f'Avg time per batch: {batch_time.avg:.3f} s\t'
          f'Avg time per sentence: {1000*(batch_time.avg / args.batch_size):.3f} ms\n'
          f'Avg enc seq len: {enc_seq_len.avg:.2f}\t'
          f'Avg dec seq len: {dec_seq_len.avg:.2f}\t'
          f'Total iterations: {total_iters}\t\n'
          f'Avg tot tok/s: {tot_tok_per_sec.avg:.0f}\t'
          f'Avg enc tok/s: {enc_tok_per_sec.avg:.0f}\t'
          f'Avg dec tok/s: {dec_tok_per_sec.avg:.0f}')

if __name__ == '__main__':
    main()
