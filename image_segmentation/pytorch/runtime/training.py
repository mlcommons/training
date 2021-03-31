from tqdm import tqdm

import torch
from torch.optim import Adam, SGD
from torch.cuda.amp import autocast, GradScaler

from runtime.distributed_utils import get_rank, reduce_tensor, get_world_size
from runtime.inference import evaluate
from runtime.logging import mllog_event, mllog_start, mllog_end, CONSTANTS

TRAIN_DATASET_SIZE = 168


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
    scale = min(current_samples / warmup_samples, 1.0)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr + (lr - init_lr) * scale


def train(flags, model, train_loader, val_loader, loss_fn, score_fn, device, callbacks, is_distributed):
    rank = get_rank()
    world_size = get_world_size()
    global_batch_size = world_size * flags.batch_size * flags.ga_steps
    next_eval_at = flags.start_eval_at

    torch.backends.cudnn.benchmark = flags.cudnn_benchmark
    torch.backends.cudnn.deterministic = flags.cudnn_deterministic

    optimizer = get_optimizer(model.parameters(), flags)
    if flags.lr_decay_samples:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=flags.lr_decay_samples,
                                                         gamma=flags.lr_decay_factor)
    scaler = GradScaler()

    model.to(device)
    loss_fn.to(device)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[flags.local_rank],
                                                          output_device=flags.local_rank)

    stop_training = False
    seen_samples = 0
    model.train()
    for callback in callbacks:
        callback.on_fit_start()
    for epoch in range(1, flags.epochs + 1):
        cumulative_loss = []
        if seen_samples <= flags.lr_warmup_samples + TRAIN_DATASET_SIZE and flags.lr_warmup_samples > 0:
            lr_warmup(optimizer, flags.init_learning_rate, flags.learning_rate, seen_samples, flags.lr_warmup_samples)
        mllog_start(key=CONSTANTS.BLOCK_START, sync=True,
                    metadata={CONSTANTS.FIRST_EPOCH_NUM: seen_samples + 1,
                              CONSTANTS.EPOCH_COUNT: global_batch_size * (TRAIN_DATASET_SIZE // global_batch_size)})
        mllog_start(key=CONSTANTS.EPOCH_START, metadata={CONSTANTS.EPOCH_NUM: seen_samples + 1}, sync=True)

        if is_distributed:
            train_loader.sampler.set_epoch(epoch)
            # val_loader.sampler.set_epoch(epoch)

        loss_value = None
        optimizer.zero_grad()
        current_samples = 0
        for iteration, batch in enumerate(tqdm(train_loader, disable=(rank != 0) or not flags.verbose)):
            image, label = batch
            image = image.view(flags.ga_steps, flags.batch_size, 1, *flags.input_shape)
            label = label.view(flags.ga_steps, flags.batch_size, 1, *flags.input_shape)
            for callback in callbacks:
                callback.on_batch_start()

            for curr_image, curr_label in zip(image, label):
                curr_image, curr_label = curr_image.to(device), curr_label.to(device)

                with autocast(enabled=flags.amp):
                    output = model(curr_image)
                    loss_value = loss_fn(output, curr_label)
                    loss_value /= flags.ga_steps

                if flags.amp:
                    scaler.scale(loss_value).backward()
                else:
                    loss_value.backward()

            if flags.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            loss_value = reduce_tensor(loss_value, world_size).detach().cpu().numpy()
            cumulative_loss.append(loss_value)
            current_samples += global_batch_size
            if flags.lr_decay_samples:
                scheduler.step()

        mllog_end(key=CONSTANTS.EPOCH_STOP, sync=True,
                  metadata={CONSTANTS.EPOCH_NUM: seen_samples + 1,
                            'current_lr': optimizer.param_groups[0]['lr'],
                            'current_loss': sum(cumulative_loss) / len(cumulative_loss)})

        seen_samples += current_samples

        if (seen_samples >= next_eval_at) and seen_samples >= flags.start_eval_at:
            del output
            next_eval_at += flags.evaluate_every
            mllog_start(key=CONSTANTS.EVAL_START, value=seen_samples, sync=True,
                        metadata={CONSTANTS.EPOCH_NUM: seen_samples})

            eval_metrics = evaluate(flags, model, val_loader, loss_fn, score_fn, device, seen_samples)
            eval_metrics["train_loss"] = sum(cumulative_loss) / len(cumulative_loss)
            eval_metrics["samples"] = seen_samples

            mllog_event(key=CONSTANTS.EVAL_ACCURACY,
                        value=eval_metrics["mean_dice"],
                        metadata={CONSTANTS.EPOCH_NUM: seen_samples},
                        sync=False)
            mllog_end(key=CONSTANTS.EVAL_STOP, metadata={CONSTANTS.EPOCH_NUM: seen_samples}, sync=True)

            for callback in callbacks:
                callback.on_epoch_end(epoch=epoch, metrics=eval_metrics, model=model, optimizer=optimizer)
            model.train()
            if eval_metrics["mean_dice"] >= flags.quality_threshold:
                stop_training = True

        mllog_end(key=CONSTANTS.BLOCK_STOP, sync=True,
                  metadata={CONSTANTS.FIRST_EPOCH_NUM: seen_samples, CONSTANTS.EPOCH_COUNT: 1})

        if stop_training:
            break

    mllog_end(key=CONSTANTS.RUN_STOP, sync=True,
              metadata={CONSTANTS.STATUS: CONSTANTS.SUCCESS if stop_training else CONSTANTS.ABORTED})
    for callback in callbacks:
        callback.on_fit_end()
