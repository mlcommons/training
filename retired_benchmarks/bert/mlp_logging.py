""" MLPerf logging utilities functions."""

import os
import absl

from mlperf_logging import mllog
from mlperf_logging.mllog import constants

flags = absl.flags
FLAGS = flags.FLAGS
CONSTANTS = constants
mllogger = mllog.get_mllogger()


def get_mlperf_logger(path, filename='mlperf.log'):
    mllog.config(filename=os.path.join(path, filename))
    mllogger = mllog.get_mllogger()
    mllogger.logger.propagate = False
    return mllogger


def mllog_start(*args, **kwargs):
    _mllog_print(mllogger.start, *args, **kwargs)


def mllog_end(*args, **kwargs):
    _mllog_print(mllogger.end, *args, **kwargs)


def mllog_event(*args, **kwargs):
    _mllog_print(mllogger.event, *args, **kwargs)


def _mllog_print(logger, *args, **kwargs):
    """
    Wrapper for MLPerf compliance logging calls.
    """
    if 'value' not in kwargs:
        kwargs['value'] = None
    logger(*args, **kwargs, stack_offset=3)


def mlperf_submission_log():
    mllog_event(
        key=mllog.constants.SUBMISSION_BENCHMARK,
        value='bert',
    )

    mllog_event(
        key=mllog.constants.SUBMISSION_ORG,
        value='reference_implementation')

    mllog_event(
        key=mllog.constants.SUBMISSION_DIVISION,
        value='closed')

    mllog_event(
        key=mllog.constants.SUBMISSION_STATUS,
        value='reference_implementation')

    mllog_event(
        key=mllog.constants.SUBMISSION_PLATFORM,
        value='reference_implementation')


def mlperf_run_param_log():
    mllog_event(key=mllog.constants.GLOBAL_BATCH_SIZE, value=FLAGS.train_batch_size)
    mllog_event(key=mllog.constants.MAX_SEQUENCE_LENGTH, value=FLAGS.max_seq_length)
    mllog_event(key='max_prediction_per_seq', value=FLAGS.max_predictions_per_seq)

    mllog_event(key=mllog.constants.OPT_NAME, value=FLAGS.optimizer)
    mllog_event(key=mllog.constants.OPT_BASE_LR, value=FLAGS.learning_rate)
    mllog_event(key=mllog.constants.OPT_WEIGHT_DECAY, value=FLAGS.lamb_weight_decay_rate)
    mllog_event(key=mllog.constants.OPT_LAMB_BETA_1, value=FLAGS.lamb_beta_1)
    mllog_event(key=mllog.constants.OPT_LAMB_BETA_2, value=FLAGS.lamb_beta_2)
    mllog_event(key=mllog.constants.OPT_LAMB_LR_DECAY_POLY_POWER, value=FLAGS.poly_power)
    mllog_event(key=mllog.constants.OPT_LAMB_EPSILON, value=10**FLAGS.log_epsilon)

    mllog_event(key=mllog.constants.OPT_LR_WARMUP_STEPS, value=FLAGS.num_warmup_steps)
    mllog_event(key='start_warmup_step', value=FLAGS.start_warmup_step)
    mllog_event(key='opt_learning_rate_trainnig_steps', value=FLAGS.num_train_steps)
    mllog_event(key=mllog.constants.GRADIENT_ACCUMULATION_STEPS, value=FLAGS.steps_per_update)
    mllog_event(key=mllog.constants.EVAL_SAMPLES, value=10000)
    mllog_event(key=mllog.constants.TRAIN_SAMPLES, value=FLAGS.train_batch_size * FLAGS.num_train_steps)


