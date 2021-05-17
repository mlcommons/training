# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Common util functions and classes used by both keras cifar and imagenet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys

from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

from tf2_common.utils.flags import core as flags_core
from tf2_common.utils.misc import keras_utils
from tf2_common.utils.mlp_log import mlp_log
import imagenet_preprocessing
import lars_optimizer
import lars_util
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2

FLAGS = flags.FLAGS
# BASE_LEARNING_RATE = 0.1  # This matches Jing's version.
TRAIN_TOP_1 = 'training_accuracy_top_1'
LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]


def learning_rate_schedule(current_epoch,
                           current_batch,
                           steps_per_epoch,
                           batch_size):
  """Handles linear scaling rule, gradual warmup, and LR decay.

  Scale learning rate at epoch boundaries provided in LR_SCHEDULE by the
  provided scaling factor.

  Args:
    current_epoch: integer, current epoch indexed from 0.
    current_batch: integer, current batch in the current epoch, indexed from 0.
    steps_per_epoch: integer, number of steps in an epoch.
    batch_size: integer, total batch sized.

  Returns:
    Adjusted learning rate.
  """
  initial_lr = FLAGS.base_learning_rate * batch_size / 256
  epoch = current_epoch + float(current_batch) / steps_per_epoch
  warmup_lr_multiplier, warmup_end_epoch = LR_SCHEDULE[0]
  if epoch < warmup_end_epoch:
    # Learning rate increases linearly per step.
    return initial_lr * warmup_lr_multiplier * epoch / warmup_end_epoch
  for mult, start_epoch in LR_SCHEDULE:
    if epoch >= start_epoch:
      learning_rate = initial_lr * mult
    else:
      break
  return learning_rate


class LearningRateBatchScheduler(tf.keras.callbacks.Callback):
  """Callback to update learning rate on every batch (not epoch boundaries).

  N.B. Only support Keras optimizers, not TF optimizers.

  Attributes:
      schedule: a function that takes an epoch index and a batch index as input
          (both integer, indexed from 0) and returns a new learning rate as
          output (float).
  """

  def __init__(self, schedule, batch_size, steps_per_epoch):
    super(LearningRateBatchScheduler, self).__init__()
    self.schedule = schedule
    self.steps_per_epoch = steps_per_epoch
    self.batch_size = batch_size
    self.epochs = -1
    self.prev_lr = -1

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'learning_rate'):
      raise ValueError('Optimizer must have a "learning_rate" attribute.')
    self.epochs += 1

  def on_batch_begin(self, batch, logs=None):
    """Executes before step begins."""
    lr = self.schedule(self.epochs,
                       batch,
                       self.steps_per_epoch,
                       self.batch_size)
    if not isinstance(lr, (float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function should be float.')
    if lr != self.prev_lr:
      self.model.optimizer.learning_rate = lr  # lr should be a float here
      self.prev_lr = lr
      tf.compat.v1.logging.debug(
          'Epoch %05d Batch %05d: LearningRateBatchScheduler '
          'change learning rate to %s.', self.epochs, batch, lr)


class PiecewiseConstantDecayWithWarmup(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Piecewise constant decay with warmup schedule."""

  def __init__(self, batch_size, steps_per_epoch, warmup_epochs, boundaries,
               multipliers, compute_lr_on_cpu=True, name=None):
    super(PiecewiseConstantDecayWithWarmup, self).__init__()
    if len(boundaries) != len(multipliers) - 1:
      raise ValueError('The length of boundaries must be 1 less than the '
                       'length of multipliers')

    base_lr_batch_size = 256
    self.steps_per_epoch = steps_per_epoch

    self.rescaled_lr = FLAGS.base_learning_rate * batch_size / base_lr_batch_size
    self.step_boundaries = [float(self.steps_per_epoch) * x for x in boundaries]
    self.lr_values = [self.rescaled_lr * m for m in multipliers]
    self.warmup_steps = warmup_epochs * self.steps_per_epoch
    self.compute_lr_on_cpu = compute_lr_on_cpu
    self.name = name

    self.learning_rate_ops_cache = {}

  def __call__(self, step):
    if tf.executing_eagerly():
      return self._get_learning_rate(step)

    # In an eager function or graph, the current implementation of optimizer
    # repeatedly call and thus create ops for the learning rate schedule. To
    # avoid this, we cache the ops if not executing eagerly.
    graph = tf.compat.v1.get_default_graph()
    if graph not in self.learning_rate_ops_cache:
      if self.compute_lr_on_cpu:
        with tf.device('/device:CPU:0'):
          self.learning_rate_ops_cache[graph] = self._get_learning_rate(step)
      else:
        self.learning_rate_ops_cache[graph] = self._get_learning_rate(step)
    return self.learning_rate_ops_cache[graph]

  def _get_learning_rate(self, step):
    """Compute learning rate at given step."""
    with tf.compat.v1.name_scope(self.name, 'PiecewiseConstantDecayWithWarmup',
                                 [self.rescaled_lr, self.step_boundaries,
                                  self.lr_values, self.warmup_steps,
                                  self.compute_lr_on_cpu]):
      def warmup_lr(step):
        return self.rescaled_lr * (
            tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32))
      def piecewise_lr(step):
        return tf.compat.v1.train.piecewise_constant(step, self.step_boundaries,
                                                     self.lr_values)

      lr = tf.cond(step < self.warmup_steps, lambda: warmup_lr(step),
                   lambda: piecewise_lr(step))
      return lr

  def get_config(self):
    return {
        'rescaled_lr': self.rescaled_lr,
        'step_boundaries': self.step_boundaries,
        'lr_values': self.lr_values,
        'warmup_steps': self.warmup_steps,
        'compute_lr_on_cpu': self.compute_lr_on_cpu,
        'name': self.name
    }


def get_optimizer(flags_obj,
                  steps_per_epoch,
                  train_steps):
  """Returns optimizer to use."""
  optimizer = None
  learning_rate_schedule_fn = None

  if (get_flag_module(flags_obj, 'model') is None or
      flags_obj.model == 'resnet50_v1.5'):
    if flags_obj.lr_schedule == 'polynomial':
      lr_schedule = lars_util.PolynomialDecayWithWarmup(
          batch_size=flags_obj.batch_size,
          steps_per_epoch=steps_per_epoch,
          train_steps=train_steps,
          initial_learning_rate=flags_obj.base_learning_rate,
          end_learning_rate=flags_obj.end_learning_rate,
          warmup_epochs=flags_obj.warmup_epochs)
    elif flags_obj.lr_schedule == 'piecewise':
      lr_schedule = PiecewiseConstantDecayWithWarmup(
          batch_size=flags_obj.batch_size,
          steps_per_epoch=steps_per_epoch,
          warmup_epochs=LR_SCHEDULE[0][1],
          boundaries=list(p[1] for p in LR_SCHEDULE[1:]),
          multipliers=list(p[0] for p in LR_SCHEDULE),
          compute_lr_on_cpu=True)
    elif flags_obj.lr_schedule == 'constant':
      lr_schedule = flags_obj.base_learning_rate * flags_obj.batch_size / 256
    else:
      raise ValueError('lr_schedule "%s" is unknown.' % flags_obj.lr_schedule)

    if flags_obj.optimizer == 'SGD':
      # The learning_rate is overwritten at the beginning of
      # each step by callback.
      optimizer = gradient_descent_v2.SGD(
          learning_rate=lr_schedule, momentum=FLAGS.momentum)

    elif flags_obj.optimizer == 'LARS':
      use_experimental_compile = True if tf.config.list_physical_devices(
          'GPU') else False

      optimizer = lars_optimizer.LARSOptimizer(
          learning_rate=lr_schedule,
          momentum=flags_obj.momentum,
          weight_decay=flags_obj.weight_decay,
          skip_list=['batch_normalization', 'bias', 'bn'],
          epsilon=flags_obj.lars_epsilon)
          # use_experimental_compile=use_experimental_compile)

    learning_rate_schedule_fn = learning_rate_schedule

  elif flags_obj.model == 'mobilenet':
    initial_learning_rate = \
          flags_obj.initial_learning_rate_per_sample * flags_obj.batch_size
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=steps_per_epoch * flags_obj.num_epochs_per_decay,
            decay_rate=flags_obj.lr_decay_factor,
            staircase=True),
        momentum=flags_obj.momentum)

  return optimizer, learning_rate_schedule_fn


# TODO(hongkuny,haoyuzhang): make cifar model use_tensor_lr to clean up code.
def get_callbacks(
    steps_per_epoch,
    learning_rate_schedule_fn=None,
    pruning_method=None,
    enable_checkpoint_and_export=False,
    model_dir=None):
  """Returns common callbacks."""
  time_callback = keras_utils.TimeHistory(
      FLAGS.batch_size,
      FLAGS.log_steps,
      logdir=FLAGS.model_dir if FLAGS.enable_tensorboard else None)
  callbacks = [time_callback]

  if FLAGS.lr_schedule == 'constant' and learning_rate_schedule_fn:
    lr_callback = LearningRateBatchScheduler(
        learning_rate_schedule_fn,
        batch_size=FLAGS.batch_size,
        steps_per_epoch=steps_per_epoch)
    callbacks.append(lr_callback)

  if FLAGS.enable_tensorboard:
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=FLAGS.model_dir)
    callbacks.append(tensorboard_callback)

  if FLAGS.profile_steps:
    profiler_callback = keras_utils.get_profiler_callback(
        FLAGS.model_dir,
        FLAGS.profile_steps,
        FLAGS.enable_tensorboard,
        steps_per_epoch)
    callbacks.append(profiler_callback)

  is_pruning_enabled = pruning_method is not None
  if is_pruning_enabled:
    callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())
    if model_dir is not None:
      callbacks.append(tfmot.sparsity.keras.PruningSummaries(
          log_dir=model_dir, profile_batch=0))

  if enable_checkpoint_and_export:
    if model_dir is not None:
      ckpt_full_path = os.path.join(model_dir, 'model.ckpt-{epoch:04d}')
      callbacks.append(
          tf.keras.callbacks.ModelCheckpoint(ckpt_full_path,
                                             save_weights_only=True))
  return callbacks


def build_stats(history, eval_output, callbacks):
  """Normalizes and returns dictionary of stats.

  Args:
    history: Results of the training step. Supports both categorical_accuracy
      and sparse_categorical_accuracy.
    eval_output: Output of the eval step. Assumes first value is eval_loss and
      second value is accuracy_top_1.
    callbacks: a list of callbacks which might include a time history callback
      used during keras.fit.

  Returns:
    Dictionary of normalized results.
  """
  stats = {}
  if eval_output:
    stats['accuracy_top_1'] = float(eval_output[1])
    if FLAGS.report_accuracy_metrics:
      stats['eval_loss'] = float(eval_output[0])

  if history and history.history and FLAGS.report_accuracy_metrics:
    train_hist = history.history
    # Gets final loss from training.
    stats['loss'] = float(train_hist['loss'][-1])
    # Gets top_1 training accuracy.
    if 'categorical_accuracy' in train_hist:
      stats[TRAIN_TOP_1] = float(train_hist['categorical_accuracy'][-1])
    elif 'sparse_categorical_accuracy' in train_hist:
      stats[TRAIN_TOP_1] = float(train_hist['sparse_categorical_accuracy'][-1])

  if not callbacks:
    return stats

  # Look for the time history callback which was used during keras.fit
  for callback in callbacks:
    if isinstance(callback, keras_utils.TimeHistory):
      timestamp_log = callback.timestamp_log
      stats['step_timestamp_log'] = timestamp_log
      stats['train_finish_time'] = callback.train_finish_time
      if callback.epoch_runtime_log:
        stats['avg_exp_per_second'] = callback.average_examples_per_second

  return stats


def define_keras_flags(
    dynamic_loss_scale=True,
    model=False,
    optimizer=False,
    pretrained_filepath=False):
  """Define flags for Keras models."""
  flags_core.define_base(clean=True, num_gpu=True, run_eagerly=True,
                         train_epochs=True, epochs_between_evals=True,
                         distribution_strategy=True)
  flags_core.define_performance(num_parallel_calls=False,
                                synthetic_data=True,
                                dtype=True,
                                all_reduce_alg=True,
                                num_packs=True,
                                tf_gpu_thread_mode=True,
                                datasets_num_private_threads=True,
                                dynamic_loss_scale=dynamic_loss_scale,
                                loss_scale=True,
                                fp16_implementation=True,
                                tf_data_experimental_slack=True,
                                enable_xla=True,
                                force_v2_in_keras_compile=True,
                                training_dataset_cache=True,
                                training_prefetch_batchs=True,
                                eval_dataset_cache=True,
                                eval_prefetch_batchs=True)
  flags_core.define_image()
  flags_core.define_benchmark()
  flags_core.define_distribution()
  flags.adopt_module_key_flags(flags_core)

  flags.DEFINE_boolean(name='enable_eager', default=False, help='Enable eager?')
  flags.DEFINE_boolean(name='skip_eval', default=False, help='Skip evaluation?')
  # TODO(b/135607288): Remove this flag once we understand the root cause of
  # slowdown when setting the learning phase in Keras backend.
  flags.DEFINE_boolean(
      name='set_learning_phase_to_train', default=True,
      help='If skip eval, also set Keras learning phase to 1 (training).')
  flags.DEFINE_boolean(
      name='explicit_gpu_placement', default=False,
      help='If not using distribution strategy, explicitly set device scope '
      'for the Keras training loop.')
  flags.DEFINE_boolean(name='use_trivial_model', default=False,
                       help='Whether to use a trivial Keras model.')
  flags.DEFINE_boolean(name='report_accuracy_metrics', default=True,
                       help='Report metrics during training and evaluation.')
  flags.DEFINE_string(
      name='lr_schedule', default='piecewise',
      help='learning rate schedule. '
      '"piecewise" for PiecewiseConstantDecayWithWarmup, '
      '"polynomial" for PolynomialDecayWithWarmup, '
      'and "constant" for static learning rate.')
  flags.DEFINE_boolean(
      name='enable_tensorboard', default=False,
      help='Whether to enable Tensorboard callback.')
  flags.DEFINE_integer(
      name='train_steps', default=None,
      help='The number of steps to run for training. If it is larger than '
      '# batches per epoch, then use # batches per epoch. This flag will be '
      'ignored if train_epochs is set to be larger than 1. ')
  flags.DEFINE_string(
      name='profile_steps', default=None,
      help='Save profiling data to model dir at given range of global steps. The '
      'value must be a comma separated pair of positive integers, specifying '
      'the first and last step to profile. For example, "--profile_steps=2,4" '
      'triggers the profiler to process 3 steps, starting from the 2nd step. '
      'Note that profiler has a non-trivial performance overhead, and the '
      'output file can be gigantic if profiling many steps.')
  flags.DEFINE_boolean(
      name='batchnorm_spatial_persistent', default=True,
      help='Enable the spacial persistent mode for CuDNN batch norm kernel.')
  flags.DEFINE_boolean(
      name='enable_get_next_as_optional', default=False,
      help='Enable get_next_as_optional behavior in DistributedIterator.')
  flags.DEFINE_boolean(
      name='enable_checkpoint_and_export', default=False,
      help='Whether to enable a checkpoint callback and export the savedmodel.')
  flags.DEFINE_string(
      name='tpu', default='', help='TPU address to connect to.')
  flags.DEFINE_string(
      name='tpu_zone', default='', help='Zone in which the TPU resides.')
  flags.DEFINE_integer(
      name='steps_per_loop',
      default=500,
      help='Number of steps per training loop. Only training step happens '
      'inside the loop. Callbacks will not be called inside. Will be capped at '
      'steps per epoch.')
  flags.DEFINE_boolean(
      name='use_tf_while_loop',
      default=True,
      help='Whether to build a tf.while_loop inside the training loop on the '
      'host. Setting it to True is critical to have peak performance on '
      'TPU.')
  flags.DEFINE_boolean(
      name='use_tf_keras_layers', default=False,
      help='Whether to use tf.keras.layers instead of tf.python.keras.layers.'
      'It only changes imagenet resnet model layers for now. This flag is '
      'a temporal flag during transition to tf.keras.layers. Do not use this '
      'flag for external usage. this will be removed shortly.')
  flags.DEFINE_float(
      'base_learning_rate', 0.1,
      'Base learning rate. '
      'This is the learning rate when using batch size 256; when using other '
      'batch sizes, the learning rate will be scaled linearly.')
  flags.DEFINE_string(
      'optimizer', 'SGD',
      'Name of optimizer preset. (SGD, LARS)')
  flags.DEFINE_boolean(
      'drop_train_remainder', True,
      'Whether to drop remainder in the training dataset.')
  flags.DEFINE_boolean(
      'drop_eval_remainder', False,
      'Whether to drop remainder in the eval dataset.')
  flags.DEFINE_float(
      'label_smoothing', 0.0,
      'Apply label smoothing to the loss. This applies to '
      'categorical_cross_entropy; when label_smoothing > 0, '
      'one-hot encoding is used for the labels.')
  flags.DEFINE_integer(
      'num_classes', 1000,
      'Number of classes for labels, at least 2.')
  flags.DEFINE_integer(
      'eval_offset_epochs', 0,
      'Epoch number of the first evaluation.')

  lars_util.define_lars_flags()

  if model:
    flags.DEFINE_string('model', 'resnet50_v1.5',
                        'Name of model preset. (mobilenet, resnet50_v1.5)')
  if optimizer:
    # TODO(kimjaehong): Replace as general hyper-params not only for mobilenet.
    flags.DEFINE_float('initial_learning_rate_per_sample', 0.00007,
                       'Initial value of learning rate per sample for '
                       'SGD optimizer when using mobilenet.')
    flags.DEFINE_float('lr_decay_factor', 0.94,
                       'Learning rate decay factor for SGD optimizer '
                       'when using mobilenet.')
    flags.DEFINE_float('num_epochs_per_decay', 2.5,
                       'Number of epochs per decay for SGD optimizer '
                       'when using mobilenet.')
  if pretrained_filepath:
    flags.DEFINE_string('pretrained_filepath', '',
                        'Pretrained file path.')
  flags.DEFINE_float('target_accuracy', 0.759,
                     'Target eval accuracy, after which training will stop.')


def get_synth_data(height, width, num_channels, num_classes, dtype):
  """Creates a set of synthetic random data.

  Args:
    height: Integer height that will be used to create a fake image tensor.
    width: Integer width that will be used to create a fake image tensor.
    num_channels: Integer depth that will be used to create a fake image tensor.
    num_classes: Number of classes that should be represented in the fake labels
      tensor
    dtype: Data type for features/images.

  Returns:
    A tuple of tensors representing the inputs and labels.

  """
  # Synthetic input should be within [0, 255].
  inputs = tf.random.truncated_normal([height, width, num_channels],
                                      dtype=dtype,
                                      mean=127,
                                      stddev=60,
                                      name='synthetic_inputs')
  labels = tf.random.uniform([1],
                             minval=0,
                             maxval=num_classes - 1,
                             dtype=tf.int32,
                             name='synthetic_labels')
  return inputs, labels


def define_pruning_flags():
  """Define flags for pruning methods."""
  flags.DEFINE_string('pruning_method', None,
                      'Pruning method.'
                      'None (no pruning) or polynomial_decay.')
  flags.DEFINE_float('pruning_initial_sparsity', 0.0,
                     'Initial sparsity for pruning.')
  flags.DEFINE_float('pruning_final_sparsity', 0.5,
                     'Final sparsity for pruning.')
  flags.DEFINE_integer('pruning_begin_step', 0,
                       'Begin step for pruning.')
  flags.DEFINE_integer('pruning_end_step', 100000,
                       'End step for pruning.')
  flags.DEFINE_integer('pruning_frequency', 100,
                       'Frequency for pruning.')


def get_synth_input_fn(height, width, num_channels, num_classes,
                       dtype=tf.float32, drop_remainder=True):
  """Returns an input function that returns a dataset with random data.

  This input_fn returns a data set that iterates over a set of random data and
  bypasses all preprocessing, e.g. jpeg decode and copy. The host to device
  copy is still included. This used to find the upper throughput bound when
  tuning the full input pipeline.

  Args:
    height: Integer height that will be used to create a fake image tensor.
    width: Integer width that will be used to create a fake image tensor.
    num_channels: Integer depth that will be used to create a fake image tensor.
    num_classes: Number of classes that should be represented in the fake labels
      tensor
    dtype: Data type for features/images.
    drop_remainder: A boolean indicates whether to drop the remainder of the
      batches. If True, the batch dimension will be static.

  Returns:
    An input_fn that can be used in place of a real one to return a dataset
    that can be used for iteration.
  """
  # pylint: disable=unused-argument
  def input_fn(is_training, data_dir, batch_size, *args, **kwargs):
    """Returns dataset filled with random data."""
    inputs, labels = get_synth_data(height=height,
                                    width=width,
                                    num_channels=num_channels,
                                    num_classes=num_classes,
                                    dtype=dtype)

    if FLAGS.label_smoothing and FLAGS.label_smoothing > 0:
      labels = tf.one_hot(labels, num_classes)
      labels = tf.reshape(labels, [num_classes])
    else:
      labels = tf.cast(labels, tf.float32)

    labels = tf.cast(labels, dtype=tf.float32)
    data = tf.data.Dataset.from_tensors((inputs, labels)).repeat()

    # `drop_remainder` will make dataset produce outputs with known shapes.
    data = data.batch(batch_size, drop_remainder=drop_remainder)
    data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return data

  return input_fn


def set_cudnn_batchnorm_mode():
  """Set CuDNN batchnorm mode for better performance.

     Note: Spatial Persistent mode may lead to accuracy losses for certain
     models.
  """
  if FLAGS.batchnorm_spatial_persistent:
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
  else:
    os.environ.pop('TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT', None)


def print_flags(flags_obj):
  """Print out all flags."""
  flags_by_module = flags_obj.flags_by_module_dict()
  modules = sorted(flags_by_module)
  main_module = sys.argv[0]
  if main_module in modules:
    modules.remove(main_module)
    modules = [main_module] + modules

  selections = ['mlperf', 'tensorflow', 'absl', 'xla', 'tf2', 'main']
  for module in modules:
    hit_selections = False
    for selection in selections:
      if selection in module:
        hit_selections = True
        break
    # if not hit_selections:
    #   continue

    logging.info('Module %s:', module)
    flags_dict = flags_by_module[module]
    for flag in flags_dict:
      logging.info('\t flags_obj.%s = %s', flag.name, flag.value)


def get_flag_module(flags_obj, flag):
  """Get which module a flag is defined in."""
  flags_by_module = flags_obj.flags_by_module_dict()
  modules = sorted(flags_by_module)

  for module in modules:
    if flag in flags_by_module[module]:
      return module

  return None


def get_num_train_iterations(flags_obj):
  """Returns the number of training steps, train and test epochs."""
  if flags_obj.drop_train_remainder:
    steps_per_epoch = (
        imagenet_preprocessing.NUM_IMAGES['train'] // flags_obj.batch_size)
  else:
    steps_per_epoch = (
        math.ceil(1.0 * imagenet_preprocessing.NUM_IMAGES['train'] /
                  flags_obj.batch_size))

  train_epochs = flags_obj.train_epochs
  # if mutliple epochs, ignore the train_steps flag.
  if train_epochs <= 1 and flags_obj.train_steps:
    steps_per_epoch = min(flags_obj.train_steps, steps_per_epoch)
    train_epochs = 1
  else:
    eval_offset_epochs = flags_obj.eval_offset_epochs
    epochs_between_evals = flags_obj.epochs_between_evals
    train_epochs = eval_offset_epochs + math.ceil(
        (train_epochs - eval_offset_epochs) /
        epochs_between_evals) * epochs_between_evals

  return steps_per_epoch, train_epochs


def get_num_eval_steps(flags_obj):
  """Returns the number of eval steps."""
  if flags_obj.drop_eval_remainder:
    eval_steps = (
        imagenet_preprocessing.NUM_IMAGES['validation'] // flags_obj.batch_size)
  else:
    eval_steps = (
        math.ceil(1.0 * imagenet_preprocessing.NUM_IMAGES['validation'] /
                  flags_obj.batch_size))

  return eval_steps

