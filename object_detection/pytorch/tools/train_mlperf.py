# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import functools
import logging
import random
import datetime
import time

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.engine.tester import test
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.mlperf_logger import print_mlperf, generate_seeds, broadcast_seeds

from mlperf_compliance import mlperf_log

def test_and_exchange_map(tester, model, distributed):
    results = tester(model=model, distributed=distributed)

    # main process only
    if is_main_process():
        # Note: one indirection due to possibility of multiple test datasets, we only care about the first
        #       tester returns (parsed results, raw results). In our case, don't care about the latter
        map_results, raw_results = results[0]
        bbox_map = map_results.results["bbox"]['AP']
        segm_map = map_results.results["segm"]['AP']
    else:
        bbox_map = 0.
        segm_map = 0.

    if distributed:
        map_tensor = torch.tensor([bbox_map, segm_map], dtype=torch.float32, device=torch.device("cuda"))
        torch.distributed.broadcast(map_tensor, 0)
        bbox_map = map_tensor[0].item()
        segm_map = map_tensor[1].item()

    return bbox_map, segm_map

def mlperf_test_early_exit(iteration, iters_per_epoch, tester, model, distributed, min_bbox_map, min_segm_map):
    # Note: let iters / epoch == 10k, at iter 9999 we've finished epoch 0 and need to test
    if iteration > 0 and (iteration + 1)% iters_per_epoch == 0:
        epoch = iteration // iters_per_epoch

        print_mlperf(key=mlperf_log.EVAL_START, value=epoch)

        bbox_map, segm_map = test_and_exchange_map(tester, model, distributed)
        # necessary for correctness
        model.train()

        print_mlperf(key=mlperf_log.EVAL_TARGET, value={"BBOX": min_bbox_map,
                                                        "SEGM": min_segm_map})
        logger = logging.getLogger('maskrcnn_benchmark.trainer')
        logger.info('bbox mAP: {}, segm mAP: {}'.format(bbox_map, segm_map))

        print_mlperf(key=mlperf_log.EVAL_ACCURACY, value={"epoch" : epoch, "value":{"BBOX" : bbox_map, "SEGM" : segm_map}})
        print_mlperf(key=mlperf_log.EVAL_STOP)

        # terminating condition
        if bbox_map >= min_bbox_map and segm_map >= min_segm_map:
            logger.info("Target mAP reached, exiting...")
            print_mlperf(key=mlperf_log.RUN_STOP, value={"success":True})
            return True

        # At this point will start the next epoch, so note this in the log
        # print_mlperf(key=mlperf_log.TRAIN_EPOCH, value=epoch+1)
    return False

def mlperf_log_epoch_start(iteration, iters_per_epoch):
    # First iteration:
    #     Note we've started training & tag first epoch start
    if iteration == 0:
        print_mlperf(key=mlperf_log.TRAIN_LOOP)
        print_mlperf(key=mlperf_log.TRAIN_EPOCH, value=0)
        return
    if iteration % iters_per_epoch == 0:
        epoch = iteration // iters_per_epoch
        print_mlperf(key=mlperf_log.TRAIN_EPOCH, value=epoch)

from maskrcnn_benchmark.layers.batch_norm import FrozenBatchNorm2d
def cast_frozen_bn_to_half(module):
    if isinstance(module, FrozenBatchNorm2d):
        module.half()
    for child in module.children():
        cast_frozen_bn_to_half(child)
    return module

def train(cfg, local_rank, distributed):
    # Model logging
    print_mlperf(key=mlperf_log.INPUT_BATCH_SIZE, value=cfg.SOLVER.IMS_PER_BATCH)
    print_mlperf(key=mlperf_log.BATCH_SIZE_TEST, value=cfg.TEST.IMS_PER_BATCH)

    print_mlperf(key=mlperf_log.INPUT_MEAN_SUBTRACTION, value = cfg.INPUT.PIXEL_MEAN)
    print_mlperf(key=mlperf_log.INPUT_NORMALIZATION_STD, value=cfg.INPUT.PIXEL_STD)
    print_mlperf(key=mlperf_log.INPUT_RESIZE)
    print_mlperf(key=mlperf_log.INPUT_RESIZE_ASPECT_PRESERVING)
    print_mlperf(key=mlperf_log.MIN_IMAGE_SIZE, value=cfg.INPUT.MIN_SIZE_TRAIN)
    print_mlperf(key=mlperf_log.MAX_IMAGE_SIZE, value=cfg.INPUT.MAX_SIZE_TRAIN)
    print_mlperf(key=mlperf_log.INPUT_RANDOM_FLIP)
    print_mlperf(key=mlperf_log.RANDOM_FLIP_PROBABILITY, value=0.5)
    print_mlperf(key=mlperf_log.FG_IOU_THRESHOLD, value=cfg.MODEL.RPN.FG_IOU_THRESHOLD)
    print_mlperf(key=mlperf_log.BG_IOU_THRESHOLD, value=cfg.MODEL.RPN.BG_IOU_THRESHOLD)
    print_mlperf(key=mlperf_log.RPN_PRE_NMS_TOP_N_TRAIN, value=cfg.MODEL.RPN.PRE_NMS_TOP_N_TRAIN)
    print_mlperf(key=mlperf_log.RPN_PRE_NMS_TOP_N_TEST, value=cfg.MODEL.RPN.PRE_NMS_TOP_N_TEST)
    print_mlperf(key=mlperf_log.RPN_POST_NMS_TOP_N_TRAIN, value=cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN)
    print_mlperf(key=mlperf_log.RPN_POST_NMS_TOP_N_TEST, value=cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST)
    print_mlperf(key=mlperf_log.ASPECT_RATIOS, value=cfg.MODEL.RPN.ASPECT_RATIOS)
    print_mlperf(key=mlperf_log.BACKBONE, value=cfg.MODEL.BACKBONE.CONV_BODY)
    print_mlperf(key=mlperf_log.NMS_THRESHOLD, value=cfg.MODEL.RPN.NMS_THRESH)

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    # Optimizer logging
    print_mlperf(key=mlperf_log.OPT_NAME, value=mlperf_log.SGD_WITH_MOMENTUM)
    print_mlperf(key=mlperf_log.OPT_LR, value=cfg.SOLVER.BASE_LR)
    print_mlperf(key=mlperf_log.OPT_MOMENTUM, value=cfg.SOLVER.MOMENTUM)
    print_mlperf(key=mlperf_log.OPT_WEIGHT_DECAY, value=cfg.SOLVER.WEIGHT_DECAY)


    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    arguments["save_checkpoints"] = cfg.SAVE_CHECKPOINTS

    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    data_loader, iters_per_epoch = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    # set the callback function to evaluate and potentially
    # early exit each epoch
    if cfg.PER_EPOCH_EVAL:
        per_iter_callback_fn = functools.partial(
                mlperf_test_early_exit,
                iters_per_epoch=iters_per_epoch,
                tester=functools.partial(test, cfg=cfg),
                model=model,
                distributed=distributed,
                min_bbox_map=cfg.MLPERF.MIN_BBOX_MAP,
                min_segm_map=cfg.MLPERF.MIN_SEGM_MAP)
    else:
        per_iter_callback_fn = None

    start_train_time = time.time()

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        per_iter_start_callback_fn=functools.partial(mlperf_log_epoch_start, iters_per_epoch=iters_per_epoch),
        per_iter_end_callback_fn=per_iter_callback_fn,
    )

    end_train_time = time.time()
    total_training_time = end_train_time - start_train_time
    print(
            "&&&& MLPERF METRIC THROUGHPUT per GPU={:.4f} iterations / s".format((arguments["iteration"] * 1.0) / total_training_time)
    )

    return model



def main():
    mlperf_log.ROOT_DIR_MASKRCNN = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if is_main_process:
        # Setting logging file parameters for compliance logging
        os.environ["COMPLIANCE_FILE"] = '/MASKRCNN_complVv0.5.0_' + str(datetime.datetime.now())
        mlperf_log.LOG_FILE = os.getenv("COMPLIANCE_FILE")
        mlperf_log._FILE_HANDLER = logging.FileHandler(mlperf_log.LOG_FILE)
        mlperf_log._FILE_HANDLER.setLevel(logging.DEBUG)
        mlperf_log.LOGGER.addHandler(mlperf_log._FILE_HANDLER)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

        print_mlperf(key=mlperf_log.RUN_START)

        # setting seeds - needs to be timed, so after RUN_START
        if is_main_process():
            master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)
            seed_tensor = torch.tensor(master_seed, dtype=torch.float32, device=torch.device("cuda"))
        else:
            seed_tensor = torch.tensor(0, dtype=torch.float32, device=torch.device("cuda"))

        torch.distributed.broadcast(seed_tensor, 0)
        master_seed = int(seed_tensor.item())
    else:
        print_mlperf(key=mlperf_log.RUN_START)
        # random master seed, random.SystemRandom() uses /dev/urandom on Unix
        master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)

    # actually use the random seed
    args.seed = master_seed
    # random number generator with seed set to master_seed
    random_number_generator = random.Random(master_seed)
    print_mlperf(key=mlperf_log.RUN_SET_RANDOM_SEED, value=master_seed)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    # generate worker seeds, one seed for every distributed worker
    worker_seeds = generate_seeds(random_number_generator, torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)

    # todo sharath what if CPU
    # broadcast seeds from rank=0 to other workers
    worker_seeds = broadcast_seeds(worker_seeds, device='cuda')

    # Setting worker seeds
    logger.info("Worker {}: Setting seed {}".format(args.local_rank, worker_seeds[args.local_rank]))
    torch.manual_seed(worker_seeds[args.local_rank])


    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args.local_rank, args.distributed)

    print_mlperf(key=mlperf_log.RUN_FINAL)


if __name__ == "__main__":
    start = time.time()
    main()
    print("&&&& MLPERF METRIC TIME=", time.time() - start)
