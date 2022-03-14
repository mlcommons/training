import os
import time
import random
import argparse
import datetime

import numpy as np
import torch
import torch.utils.data
import torchvision

from ssd_logger import mllogger
from mlperf_logging.mllog.constants import (SUBMISSION_BENCHMARK, SUBMISSION_DIVISION, SUBMISSION_STATUS,
    SSD, CLOSED, ONPREM, EVAL_ACCURACY, STATUS, SUCCESS, ABORTED,
    INIT_START, INIT_STOP, RUN_START, RUN_STOP, SEED, GLOBAL_BATCH_SIZE, TRAIN_SAMPLES,
    EVAL_SAMPLES, EPOCH_COUNT, FIRST_EPOCH_NUM, OPT_NAME, ADAM, OPT_BASE_LR, OPT_WEIGHT_DECAY,
    OPT_LR_WARMUP_EPOCHS, OPT_LR_WARMUP_FACTOR, GRADIENT_ACCUMULATION_STEPS)

import utils
import presets
from coco_utils import get_coco, get_openimages
from engine import train_one_epoch, evaluate
from model.retinanet import retinanet_from_backbone



def get_dataset_fn(name):
    paths = {
        "coco": (get_coco, 91),
        "openimages": (get_openimages, 601),            # Full openimages dataset
        "openimages-mlperf": (get_openimages, 264),     # L0 classes with more than 1000 samples
    }
    return paths[name]


def get_transform(train, data_augmentation):
    return presets.DetectionPresetTrain(data_augmentation) if train else presets.DetectionPresetEval()


def parse_args(add_help=True):
    parser = argparse.ArgumentParser(description='PyTorch Detection Training', add_help=add_help)

    # Model
    parser.add_argument('--backbone', default='resnext50_32x4d',
                        choices=['resnet50', 'resnext50_32x4d', 'resnet101', 'resnext101_32x8d'],
                        help='The model backbone')
    parser.add_argument('--trainable-backbone-layers', default=3, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument('--sync-bn', dest='sync_bn', action="store_true", help='Use sync batch norm')
    parser.add_argument('--data-layout', default="channels_last", choices=['channels_first', 'channels_last'],
                        help="Model data layout")
    parser.add_argument("--amp", dest='amp', action="store_true",
                        help="Whether to enable Automatic Mixed Precision (AMP). "
                             "When false, uses TF32 on A100 and FP32 on V100 GPUS.")
    parser.add_argument("--no-amp", dest='amp', action="store_false",
                        help="Whether to enable Automatic Mixed Precision (AMP). "
                             "When false, uses TF32 on A100 and FP32 on V100 GPUS.")
    parser.set_defaults(amp=True)

    # Dataset
    parser.add_argument('--dataset', default='openimages-mlperf',
                        choices=['coco', 'openimages', 'openimages-mlperf'],
                        help='dataset')
    parser.add_argument('--data-path', default='/datasets/open-images-v6', help='dataset')
    parser.add_argument('--image-size', default=[800, 800], nargs=2, type=int,
                        help='Image size for training')
    parser.add_argument('--data-augmentation', default="hflip", help='data augmentation policy')

    # Train parameters
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--output-dir', default=None, help='path where to save checkpoints.')
    parser.add_argument('--target-map', default=0.34, type=float, help='Stop training when target mAP is reached')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                        help="Use pre-trained models from the modelzoo")

    # Hyperparameters
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('-e', '--eval-batch-size', default=None, type=int,
                        help='evaluation images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--warmup-epochs', default=1, type=int,
                        help='how long the learning rate will be warmed up in fraction of epochs')
    parser.add_argument('--warmup-factor', default=1e-3, type=float,
                        help='factor for controlling warmup curve')

    # Other
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--eval-print-freq', default=None, type=int, help='eval print frequency')
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="Only test the model")
    parser.add_argument('--seed', '-s', type=int, default=random.SystemRandom().randint(0, 2**32 - 1),
                        help='manually set random seed')
    parser.add_argument('--device', default='cuda', help='device')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    args.eval_batch_size = args.eval_batch_size or args.batch_size
    args.eval_print_freq = args.eval_print_freq or args.print_freq

    return args


def main(args):
    # Init distributed mode
    utils.init_distributed_mode(args)

    # Start MLPerf benchmark
    mllogger.event(key=SUBMISSION_BENCHMARK, value=SSD)
    mllogger.event(key=SUBMISSION_DIVISION, value=CLOSED)
    mllogger.event(key=SUBMISSION_STATUS, value=ONPREM)
    mllogger.start(key=INIT_START, sync=True)

    if args.output_dir:
        utils.mkdir(args.output_dir)

    device = torch.device(args.device)

    # set rank seeds according to MLPerf rules
    if args.distributed:
        args.seed = utils.broadcast(args.seed, src=1)
        args.seed = (args.seed + utils.get_rank()) % 2**32
    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)
    mllogger.event(key=SEED, value=args.seed, log_rank=True)

    # Print args
    mllogger.event(key='local_batch_size', value=args.batch_size)
    mllogger.event(key=GLOBAL_BATCH_SIZE, value=args.batch_size*utils.get_world_size())
    mllogger.event(key=EPOCH_COUNT, value=args.epochs)
    mllogger.event(key=FIRST_EPOCH_NUM, value=args.start_epoch)
    print(args)

    # Data loading code
    print("Getting dataset information")
    dataset_fn, num_classes = get_dataset_fn(name=args.dataset)
    args.num_classes = num_classes

    print("Creating model")
    model = retinanet_from_backbone(backbone=args.backbone,
                                    num_classes=num_classes,
                                    image_size=args.image_size,
                                    data_layout=args.data_layout,
                                    pretrained=args.pretrained,
                                    trainable_backbone_layers=args.trainable_backbone_layers)
    model.to(device)

    if args.data_layout == 'channels_last':
        model = model.to(memory_format=torch.channels_last)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    mllogger.event(key=OPT_NAME, value=ADAM)
    mllogger.event(key=OPT_BASE_LR, value=args.lr)
    mllogger.event(key=OPT_WEIGHT_DECAY, value=0)
    mllogger.event(key=OPT_LR_WARMUP_EPOCHS, value=args.warmup_epochs)
    mllogger.event(key=OPT_LR_WARMUP_FACTOR, value=args.warmup_factor)
    mllogger.event(key=GRADIENT_ACCUMULATION_STEPS, value=1)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1

    # GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    mllogger.end(key=INIT_STOP, sync=True)

    status = ABORTED
    start_time = time.time()
    mllogger.start(key=RUN_START, sync=True)

    # We can't touch data before RUN_START
    print("Creating data loaders")
    dataset = dataset_fn(name=args.dataset,
                         root=args.data_path,
                         image_set="train",
                         transforms=get_transform(True, args.data_augmentation))
    dataset_test = dataset_fn(name=args.dataset,
                              root=args.data_path,
                              image_set="val",
                              transforms=get_transform(False, args.data_augmentation))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.eval_batch_size or args.batch_size,
        sampler=test_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=utils.collate_fn)
    mllogger.event(key=TRAIN_SAMPLES, value=len(data_loader))
    mllogger.event(key=EVAL_SAMPLES, value=len(data_loader_test))

    print("Running ...")
    status = ABORTED
    if args.test_only:
        coco_evaluator = evaluate(model, data_loader_test, device=device, epoch=None, args=args)
        accuracy = coco_evaluator.get_stats()['bbox'][0]
        print(f'Model mAP = {accuracy}')
        if args.target_map and accuracy >= args.target_map:
            status = SUCCESS
    else:
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            train_one_epoch(model, optimizer, scaler, data_loader, device, epoch, args)
            if args.output_dir:
                checkpoint = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'checkpoint.pth'))

            # evaluate after every epoch
            coco_evaluator = evaluate(model, data_loader_test, device=device, epoch=epoch, args=args)
            accuracy = coco_evaluator.get_stats()['bbox'][0]
            if args.target_map and accuracy >= args.target_map:
                status = SUCCESS
                break

    mllogger.end(key=RUN_STOP, metadata={"status": status}, sync=True)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    mllogger.event(key=STATUS, value=status, log_rank=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
