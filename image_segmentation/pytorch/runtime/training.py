from tqdm import tqdm
from time import time

import torch
from torch.optim import Adam, SGD
from torch.cuda.amp import autocast, GradScaler

from runtime.distributed_utils import get_rank, reduce_tensor, get_world_size
from runtime.inference import evaluate
from runtime.logging import mllog_event, mllog_start, mllog_end, CONSTANTS


START_EVAL_AT = 168*1000
EVALUATE_EVERY = 168*20


def get_optimizer(params, flags):
    if flags.optimizer == "adam":
        optim = Adam(params, lr=flags.learning_rate, weight_decay=flags.weight_decay)
    elif flags.optimizer == "sgd":
        optim = SGD(params, lr=flags.learning_rate, momentum=flags.momentum, nesterov=True,
                    weight_decay=flags.weight_decay)
    elif flags.optimizer == "lamb":
        import apex
        optim = apex.optimizers.FusedLAMB(params, lr=flags.learning_rate, betas=flags.lamb_betas,
                                          weight_decay=flags.weight_decay)
    else:
        raise ValueError("Optimizer {} unknown.".format(flags.optimizer))
    return optim


def lr_warmup(optimizer, init_lr, lr, current_samples, warmup_samples):
    scale = current_samples / warmup_samples
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr + (lr - init_lr) * scale


def lr_decay(optimizer, lr_decay_samples, lr_decay_factor, total_samples):
    if len(lr_decay_samples) > 0 and total_samples > lr_decay_samples[0]:
        lr_decay_samples = lr_decay_samples[1:]
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay_factor
    return lr_decay_samples


def train(flags, model, train_loader, val_loader, loss_fn, score_fn, device, callbacks, is_distributed):

    world_size = get_world_size()
    torch.backends.cudnn.benchmark = flags.cudnn_benchmark
    torch.backends.cudnn.deterministic = flags.cudnn_deterministic

    optimizer = get_optimizer(model.parameters(), flags)
    scaler = GradScaler()

    model.to(device)
    loss_fn.to(device)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[flags.local_rank],
                                                          output_device=flags.local_rank)

    is_successful = False
    diverged = False
    total_samples = 0
    iteration = 0
    lr_decay_samples = flags.lr_decay_samples
    next_eval_at = EVALUATE_EVERY
    model.train()
    train_loader = iter(train_loader)
    for callback in callbacks:
        callback.on_fit_start()

    while not diverged and not is_successful:
        mllog_start(key=CONSTANTS.BLOCK_START, sync=False,
                    metadata={CONSTANTS.FIRST_EPOCH_NUM: total_samples,
                              CONSTANTS.EPOCH_COUNT: EVALUATE_EVERY})

        t0 = time()
        while total_samples < next_eval_at:
            if total_samples <= flags.lr_warmup_samples and flags.lr_warmup_samples > 0:
                lr_warmup(optimizer, flags.init_learning_rate, flags.learning_rate, total_samples, flags.lr_warmup_samples)
            if len(flags.lr_decay_samples) > 0:
                lr_decay_samples = lr_decay(optimizer, lr_decay_samples, flags.lr_decay_factor, total_samples)

            optimizer.zero_grad()

            batch = next(train_loader)
            total_samples += flags.batch_size * world_size

            image, label = batch
            image, label = image.to(device), label.to(device)
            for callback in callbacks:
                callback.on_batch_start()

            with autocast(enabled=flags.amp):
                output = model(image)
                loss_value = loss_fn(output, label)
                loss_value /= flags.ga_steps

            if flags.amp:
                scaler.scale(loss_value).backward()
            else:
                loss_value.backward()

            if (iteration + 1) % flags.ga_steps == 0:
                if flags.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
            iteration += 1

        # Evaluation
        del output
        if total_samples >= START_EVAL_AT:
            mllog_start(key=CONSTANTS.EVAL_START, value=total_samples,
                        metadata={CONSTANTS.EPOCH_NUM: total_samples}, sync=False)

            eval_metrics = evaluate(flags, model, val_loader, loss_fn, score_fn, device, total_samples)

            mllog_event(key=CONSTANTS.EVAL_ACCURACY, value=eval_metrics["mean_dice"],
                        metadata={CONSTANTS.EPOCH_NUM: total_samples}, sync=False)
            mllog_end(key=CONSTANTS.EVAL_STOP, metadata={CONSTANTS.EPOCH_NUM: total_samples}, sync=False)

            model.train()
            if eval_metrics["mean_dice"] >= flags.quality_threshold:
                is_successful = True
            elif eval_metrics["mean_dice"] < 1e-6:
                print("MODEL DIVERGED. ABORTING.")
                diverged = True

        mllog_end(key=CONSTANTS.BLOCK_STOP, sync=False,
                  metadata={CONSTANTS.FIRST_EPOCH_NUM: total_samples,
                            CONSTANTS.EPOCH_COUNT: EVALUATE_EVERY})
        next_eval_at += EVALUATE_EVERY

    mllog_end(key=CONSTANTS.RUN_STOP, sync=True,
              metadata={CONSTANTS.STATUS: CONSTANTS.SUCCESS if is_successful else CONSTANTS.ABORTED,
                        CONSTANTS.EPOCH_COUNT: total_samples})

    for callback in callbacks:
        callback.on_fit_end()
