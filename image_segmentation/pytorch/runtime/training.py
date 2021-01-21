import numpy as np
from tqdm import tqdm

# import apex
import torch
from torch.optim import Adam, SGD
from torch.cuda.amp import autocast, GradScaler

from runtime.distributed_utils import get_rank, reduce_tensor, get_world_size
from runtime.inference import evaluate
from runtime.logging import mllog_event, mllog_start, mllog_end, CONSTANTS


EVAL_EPOCHS = [100]
EVAL_EPOCHS.extend([i for i in range(1000, 4000, 1000)])  # 3
EVAL_EPOCHS.extend([i for i in range(3100, 4100, 100)])  # 10
EVAL_EPOCHS.extend([i for i in range(4025, 5025, 25)])  # 40
EVAL_EPOCHS.extend([i for i in range(5010, 6010, 10)])  # 100

HITS_REQUIRED = 3
DICE_THRESHOLD = 0.91


def get_optimizer(params, flags):
    if flags.optimizer == "adam":
        optim = Adam(params, lr=flags.learning_rate, weight_decay=flags.weight_decay)
    elif flags.optimizer == "sgd":
        optim = SGD(params, lr=flags.learning_rate, momentum=flags.momentum, nesterov=True,
                    weight_decay=flags.weight_decay)
    # elif flags.optimizer == "lamb":
    #     optim = apex.optimizers.FusedLAMB(params, lr=flags.learning_rate, betas=flags.lamb_betas,
    #                                       weight_decay=flags.weight_decay)
    else:
        raise ValueError("Optimizer {} unknown.".format(flags.optimizer))
    return optim


def lr_warmup(optimizer, init_lr, lr, current_epoch, warmup_epochs):
    scale = current_epoch / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr + (lr - init_lr) * scale


def train(flags, model, train_loader, val_loader, loss_fn, score_fn, device, callbacks, is_distributed):
    rank = get_rank()
    world_size = get_world_size()
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    optimizer = get_optimizer(model.parameters(), flags)
    if flags.lr_decay_epochs:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=flags.lr_decay_epochs,
                                                         gamma=flags.lr_decay_factor)
    scaler = GradScaler()

    model.to(device)
    loss_fn.to(device)
    if flags.normalization == "syncbatchnorm":
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[flags.local_rank],
                                                          output_device=flags.local_rank)

    stop_training = False
    consecutive_hits = 0
    eval_epochs = EVAL_EPOCHS
    for epoch in range(1, flags.epochs + 1):
        cumulative_loss = []
        if epoch <= flags.lr_warmup_epochs and flags.lr_warmup_epochs > 0:
            lr_warmup(optimizer, flags.init_learning_rate, flags.learning_rate, epoch, flags.lr_warmup_epochs)
        mllog_start(key=CONSTANTS.BLOCK_START, sync=True,
                    metadata={CONSTANTS.FIRST_EPOCH_NUM: epoch, CONSTANTS.EPOCH_COUNT: 1})
        mllog_start(key=CONSTANTS.EPOCH_START, metadata={CONSTANTS.EPOCH_NUM: epoch}, sync=True)
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        optimizer.zero_grad()
        accumulated_steps = 0
        for i, batch in enumerate(tqdm(train_loader, disable=(rank != 0) or not flags.verbose)):
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

            accumulated_steps += 1
            if accumulated_steps % flags.ga_steps == 0:
                if flags.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                accumulated_steps = 0

            loss_value = reduce_tensor(loss_value, world_size).detach().cpu().numpy()
            cumulative_loss.append(loss_value)

        mllog_end(key=CONSTANTS.EPOCH_STOP, sync=True,
                  metadata={CONSTANTS.EPOCH_NUM: epoch, 'current_lr': optimizer.param_groups[0]['lr']})

        if flags.lr_decay_epochs:
            scheduler.step()
        if ((epoch % flags.evaluate_every) == 0) and not flags.benchmark:
        # if (epoch in eval_epochs) and not flags.benchmark:
            del output
            mllog_start(key=CONSTANTS.EVAL_START, value=epoch, metadata={CONSTANTS.EPOCH_NUM: epoch}, sync=True)

            eval_metrics = evaluate(flags, model, val_loader, loss_fn, score_fn, device, epoch)
            eval_metrics["train_loss"] = round(sum(cumulative_loss) / len(cumulative_loss), 4)

            mllog_event(key=CONSTANTS.EVAL_ACCURACY,
                        value={"epoch": epoch, "value": round(eval_metrics["mean_dice"], 3)},
                        metadata={CONSTANTS.EPOCH_NUM: epoch},
                        sync=False)
            mllog_end(key=CONSTANTS.EVAL_STOP, metadata={CONSTANTS.EPOCH_NUM: epoch}, sync=True)

            if eval_metrics["mean_dice"] > DICE_THRESHOLD:
                consecutive_hits += 1
                eval_epochs = range(epoch, flags.epochs, 5)
            else:
                consecutive_hits = 0

            eval_metrics["consecutive_hits"] = consecutive_hits
            for callback in callbacks:
                callback.on_epoch_end(epoch, eval_metrics, model, optimizer)
            model.train()

        mllog_end(key=CONSTANTS.BLOCK_STOP, sync=True,
                  metadata={CONSTANTS.FIRST_EPOCH_NUM: epoch, CONSTANTS.EPOCH_COUNT: 1})

        if consecutive_hits == HITS_REQUIRED:
            stop_training = True
            # break

    mllog_end(key=CONSTANTS.RUN_STOP, sync=True,
              metadata={CONSTANTS.STATUS: CONSTANTS.SUCCESS if stop_training else CONSTANTS.ABORTED})
    for callback in callbacks:
        callback.on_fit_end()
