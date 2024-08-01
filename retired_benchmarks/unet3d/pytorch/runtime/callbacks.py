import time
import copy

import torch
import numpy as np


def process_performance_stats(timestamps, batch_size, mode):
    """ Get confidence intervals

    :param timestamps: Collection of timestamps
    :param batch_size: Number of samples per batch
    :param mode: Estimator's execution mode
    :return: Stats
    """
    timestamps_ms = 1000 * timestamps
    throughput_imgps = (1000.0 * batch_size / timestamps_ms).mean()
    stats = {f"throughput_{mode}": throughput_imgps,
             f"latency_{mode}_mean": timestamps_ms.mean()}
    for level in [90, 95, 99]:
        stats.update({f"latency_{mode}_{level}": np.percentile(timestamps_ms, level)})

    return stats


def get_callbacks(flags, logger, local_rank, world_size):
    callbacks = []
    if local_rank == 0:
        if not flags.benchmark:
            callbacks.append(EvaluationCallback(logger, metric="mean_dice", seed=flags.seed,
                                                threshold=flags.quality_threshold))
            if flags.save_ckpt_path:
                callbacks.append(CheckpointCallback(flags.save_ckpt_path, metric="mean_dice", seed=flags.seed))
        else:
            callbacks.append(
                PerformanceCallback(logger, flags.batch_size * world_size * flags.ga_steps,
                                    flags.warmup_steps, mode='train'))

    return callbacks


class BaseCallback:
    def on_fit_start(self, **kwargs):
        pass

    def on_batch_start(self, **kwargs):
        pass

    def on_epoch_end(self, **kwargs):
        pass

    def on_fit_end(self, **kwargs):
        pass


class PerformanceCallback(BaseCallback):
    def __init__(self, logger, batch_size, warmup_steps=20, mode='train'):
        self._logger = logger
        self._batch_size = batch_size
        self._warmup_steps = warmup_steps
        self._step = 0
        self._timestamps = []
        self._mode = mode

    def on_batch_start(self, *args, **kwargs):
        self._step += 1
        if self._step >= self._warmup_steps:
            self._timestamps.append(time.time())

    def on_fit_end(self, *args, **kwargs):
        deltas = np.array([self._timestamps[i + 1] - self._timestamps[i] for i in range(len(self._timestamps) - 1)])
        stats = process_performance_stats(deltas, self._batch_size, self._mode)

        self._logger.log(step=(), data=stats)
        self._logger.flush()


class EvaluationCallback(BaseCallback):
    def __init__(self, logger, metric, threshold=0.91, seed=0):
        self._logger = logger
        self._best_metrics = {}
        self._initialized = False
        self._main_metric = metric
        self._prefix = "TOP_"
        self._last_epoch = 0
        self._first_epoch_above_threshold = 0
        self._threshold = threshold
        self._seed = seed
        self._training_start_time = 0

    def on_fit_start(self, **kwargs):
        self._training_start_time = time.time()

    def on_epoch_end(self, epoch, metrics, *args, **kwargs):
        if not self._initialized:
            self._register_metrics(metrics)
        if self._best_metrics[self._prefix + self._main_metric] < metrics[self._main_metric]:
            for key in metrics.keys():
                self._best_metrics[self._prefix + key] = float(metrics[key])

        if metrics[self._main_metric] >= self._threshold and self._first_epoch_above_threshold == 0:
            self._first_epoch_above_threshold = epoch

        for key in metrics.keys():
            metrics[key] = float(metrics[key])
        self._last_epoch = epoch
        self._logger.log(step=(metrics["epoch"]), data={**metrics, **self._best_metrics})
        self._logger.flush()

    def _register_metrics(self, metrics):
        for key in metrics.keys():
            self._best_metrics[self._prefix + key] = float(metrics[key])
        self._initialized = True

    def on_fit_end(self, **kwargs):
        self._best_metrics["last_epoch"] = self._last_epoch
        self._best_metrics["first_conv_ep"] = self._first_epoch_above_threshold
        self._best_metrics["seed"] = self._seed
        self._best_metrics["total_time"] = (time.time() - self._training_start_time) / 60
        self._logger.log(step=(), data=self._best_metrics)
        self._logger.flush()


class CheckpointCallback(BaseCallback):
    def __init__(self, path, metric, seed):
        self._path = path
        self._main_metric = metric
        self._best_metric = 0.0
        self._best_state = {}
        self._last_state = {}
        self._seed = seed

    def on_epoch_end(self, epoch, metrics, model, optimizer, *args, **kwargs):
        try:
            current_state_dict = model.module.state_dict()
        except torch.nn.modules.module.ModuleAttributeError:
            current_state_dict = model.state_dict()
        self._last_state = {'last_model_state_dict': current_state_dict,
                            'last_optimizer_state_dict': optimizer.state_dict()}
        if metrics[self._main_metric] > self._best_metric:
            self._best_metric = metrics[self._main_metric]
            self._best_state = {'best_model_state_dict': copy.deepcopy(current_state_dict),
                                'best_optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                                **metrics}

    def on_fit_end(self, *args, **kwargs):
        torch.save({**self._last_state, **self._best_state, "seed": self._seed}, self._path)
