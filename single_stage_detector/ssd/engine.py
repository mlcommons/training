import math
import sys
import time
import torch

from ssd_logger import mllogger
from mlperf_logging.mllog.constants import (EPOCH_START, EPOCH_STOP, EVAL_START, EVAL_STOP, EVAL_ACCURACY)

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils


def train_one_epoch(model, optimizer, scaler, data_loader, device, epoch, args):
    mllogger.start(key=EPOCH_START, value=epoch, metadata={"epoch_num": epoch}, sync=True)
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch < args.warmup_epochs:
        # Convert epochs to iterations
        # we want to control warmup at the epoch level, but update lr every iteration
        start_iter = epoch*len(data_loader)
        warmup_iters = args.warmup_epochs*len(data_loader)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, start_iter, warmup_iters, args.warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, args.print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


        with torch.cuda.amp.autocast(enabled=args.amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    mllogger.end(key=EPOCH_STOP, value=epoch, metadata={"epoch_num": epoch}, sync=True)
    return metric_logger


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, args):
    mllogger.start(key=EVAL_START, value=epoch, metadata={"epoch_num": epoch}, sync=True)

    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, args.eval_print_freq, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    accuracy = coco_evaluator.get_stats()['bbox'][0]
    torch.set_num_threads(n_threads)
    mllogger.event(key=EVAL_ACCURACY, value=accuracy, metadata={"epoch_num": epoch}, clear_line=True)
    mllogger.end(key=EVAL_STOP, value=epoch, metadata={"epoch_num": epoch}, sync=True)
    return coco_evaluator
