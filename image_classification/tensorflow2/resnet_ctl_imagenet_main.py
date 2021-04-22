# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the ImageNet dataset using custom training loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from tf2_common.modeling import performance
from tf2_common.training import controller
from tf2_common.utils.flags import core as flags_core
from tf2_common.utils.logs import logger
from tf2_common.utils.misc import distribution_utils
from tf2_common.utils.misc import keras_utils
from tf2_common.utils.misc import model_helpers
from tf2_common.utils.mlp_log import mlp_log
import common
import imagenet_preprocessing
import resnet_runnable

flags.DEFINE_boolean(name='use_tf_function', default=True,
                     help='Wrap the train and test step inside a '
                     'tf.function.')
flags.DEFINE_boolean(name='single_l2_loss_op', default=False,
                     help='Calculate L2_loss on concatenated weights, '
                     'instead of using Keras per-layer L2 loss.')
flags.DEFINE_boolean(name='cache_decoded_image', default=False,
                     help='Whether or not to cache decoded images in the '
                     'input pipeline. If this flag and `cache` is enabled, '
                     'then TFExample protos will be parsed and then cached '
                     'which reduces the load on hosts.')
flags.DEFINE_boolean(name='enable_device_warmup', default=False,
                     help='Whether or not to enable device warmup. This '
                     'includes training on dummy data and enabling graph/XLA '
                     'compilation before run_start.')
flags.DEFINE_integer(name='device_warmup_steps', default=1,
                     help='The number of steps to apply for device warmup.')
flags.DEFINE_integer(name='num_replicas', default=32,
                     help='The number of TPU cores to use, '
                     'for log printout only.')


def build_stats(runnable, time_callback):
  """Normalizes and returns dictionary of stats.

  Args:
    runnable: The module containing all the training and evaluation metrics.
    time_callback: Time tracking callback instance.

  Returns:
    Dictionary of normalized results.
  """
  stats = {}

  if not runnable.flags_obj.skip_eval:
    if runnable.test_loss:
      stats['eval_loss'] = runnable.test_loss.result().numpy()
    if runnable.test_accuracy:
      stats['eval_acc'] = runnable.test_accuracy.result().numpy()

    if runnable.train_loss:
      stats['train_loss'] = runnable.train_loss.result().numpy()
    if runnable.train_accuracy:
      stats['train_acc'] = runnable.train_accuracy.result().numpy()

  if time_callback:
    timestamp_log = time_callback.timestamp_log
    stats['step_timestamp_log'] = timestamp_log
    stats['train_finish_time'] = time_callback.train_finish_time
    if time_callback.epoch_runtime_log:
      stats['avg_exp_per_second'] = time_callback.average_examples_per_second

  return stats


def _steps_to_run(steps_in_current_epoch, steps_per_epoch, steps_per_loop):
  """Calculates steps to run on device."""
  if steps_per_loop <= 0:
    raise ValueError('steps_per_loop should be positive integer.')
  if steps_per_loop == 1:
    return steps_per_loop
  return min(steps_per_loop, steps_per_epoch - steps_in_current_epoch)


def run(flags_obj):
  """Run ResNet ImageNet training and eval loop using custom training loops.

  Args:
    flags_obj: An object containing parsed flag values.

  Raises:
    ValueError: If fp16 is passed as it is not currently supported.

  Returns:
    Dictionary of training and eval stats.
  """
  mlp_log.mlperf_print('cache_clear', True)
  mlp_log.mlperf_print('init_start', None)
  mlp_log.mlperf_print('submission_benchmark', 'resnet')
  mlp_log.mlperf_print('submission_division', 'closed')
  mlp_log.mlperf_print('submission_org', 'google')
  mlp_log.mlperf_print(
      'submission_platform', 'tpu-v3-{}'.format(flags_obj.num_replicas)
      if flags_obj.tpu else 'gpu-v100-{}'.format(flags_obj.num_gpus))
  mlp_log.mlperf_print('submission_status', 'cloud')

  common.print_flags(flags_obj)

  keras_utils.set_session_config(
      enable_eager=flags_obj.enable_eager,
      enable_xla=flags_obj.enable_xla)
  performance.set_mixed_precision_policy(flags_core.get_tf_dtype(flags_obj))

  if tf.config.list_physical_devices('GPU'):
    if flags_obj.tf_gpu_thread_mode:
      datasets_num_private_threads = keras_utils.set_gpu_thread_mode_and_count(
          per_gpu_thread_count=flags_obj.per_gpu_thread_count,
          gpu_thread_mode=flags_obj.tf_gpu_thread_mode,
          num_gpus=flags_obj.num_gpus)
      if not flags_obj.datasets_num_private_threads:
        flags_obj.datasets_num_private_threads = datasets_num_private_threads
    common.set_cudnn_batchnorm_mode()

  # TODO(anj-s): Set data_format without using Keras.
  data_format = flags_obj.data_format
  if data_format is None:
    data_format = ('channels_first'
                   if tf.test.is_built_with_cuda() else 'channels_last')
  tf.keras.backend.set_image_data_format(data_format)

  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_obj.num_gpus,
      all_reduce_alg=flags_obj.all_reduce_alg,
      num_packs=flags_obj.num_packs,
      tpu_address=flags_obj.tpu,
      tpu_zone=flags_obj.tpu_zone if flags_obj.tpu else None)
  mlp_log.mlperf_print('global_batch_size', flags_obj.batch_size)
  mlp_log.mlperf_print('train_samples',
                       imagenet_preprocessing.NUM_IMAGES['train'])
  mlp_log.mlperf_print('eval_samples',
                       imagenet_preprocessing.NUM_IMAGES['validation'])
  mlp_log.mlperf_print(
      'model_bn_span',
      int(flags_obj.batch_size /
          (flags_obj.num_replicas if flags_obj.tpu else flags_obj.num_gpus)))

  per_epoch_steps, train_epochs = common.get_num_train_iterations(flags_obj)
  eval_steps = common.get_num_eval_steps(flags_obj)
  steps_per_loop = min(flags_obj.steps_per_loop, per_epoch_steps)

  logging.info(
      'Training %d epochs, each epoch has %d steps, '
      'total steps: %d; Eval %d steps', train_epochs, per_epoch_steps,
      train_epochs * per_epoch_steps, eval_steps)

  time_callback = keras_utils.TimeHistory(
      flags_obj.batch_size,
      flags_obj.log_steps,
      logdir=flags_obj.model_dir if flags_obj.enable_tensorboard else None)
  with distribution_utils.get_strategy_scope(strategy):
    runnable = resnet_runnable.ResnetRunnable(flags_obj, time_callback)

  eval_interval = (
      flags_obj.epochs_between_evals *
      per_epoch_steps if not flags_obj.skip_eval else None)
  eval_offset = (
      flags_obj.eval_offset_epochs *
      per_epoch_steps if not flags_obj.skip_eval else 0)
  if eval_offset != 0:
    eval_offset -= eval_interval
  checkpoint_interval = (
      per_epoch_steps if flags_obj.enable_checkpoint_and_export else None)
  summary_interval = per_epoch_steps if flags_obj.enable_tensorboard else None

  checkpoint_manager = tf.train.CheckpointManager(
      runnable.checkpoint,
      directory=flags_obj.model_dir,
      max_to_keep=10,
      step_counter=runnable.global_step,
      checkpoint_interval=checkpoint_interval)

  device_warmup_steps = (flags_obj.device_warmup_steps
                         if flags_obj.enable_device_warmup else 0)
  if flags_obj.enable_device_warmup:
    logging.info('Warmup for %d steps.', device_warmup_steps)

  resnet_controller = controller.Controller(
      strategy,
      runnable.train,
      runnable.evaluate,
      runnable.warmup,
      global_step=runnable.global_step,
      steps_per_loop=steps_per_loop,
      train_steps=per_epoch_steps * train_epochs,
      device_warmup_steps=device_warmup_steps,
      checkpoint_manager=checkpoint_manager,
      summary_interval=summary_interval,
      eval_steps=eval_steps,
      eval_interval=eval_interval,
      eval_offset=eval_offset)

  if flags_obj.enable_device_warmup:
    resnet_controller.warmup()

  mlp_log.mlperf_print('init_stop', None)

  profile_steps = flags_obj.profile_steps
  if profile_steps:
    profile_steps = [int(i) for i in profile_steps.split(',')]
    if profile_steps[0] < 0:
      runnable.trace_start(-1)

  time_callback.on_train_begin()
  mlp_log.mlperf_print('run_start', None)
  mlp_log.mlperf_print(
      'block_start',
      None,
      metadata={
          'first_epoch_num':
              1,
          'epoch_count':
              (flags_obj.eval_offset_epochs if flags_obj.eval_offset_epochs != 0
               else flags_obj.epochs_between_evals)
      })
  resnet_controller.train(evaluate=not flags_obj.skip_eval)
  mlp_log.mlperf_print('run_stop', None, metadata={'status': 'success'})
  time_callback.on_train_end()
  mlp_log.mlperf_print('run_final', None)

  stats = build_stats(runnable, time_callback)
  return stats


def define_imagenet_keras_flags():
  common.define_keras_flags()
  flags_core.set_defaults()
  flags.adopt_module_key_flags(common)


def main(_):
  model_helpers.apply_clean(flags.FLAGS)
  with logger.benchmark_context(flags.FLAGS):
    stats = run(flags.FLAGS)
  logging.info('Run stats:\n%s', stats)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  common.define_keras_flags()
  app.run(main)
