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

from absl import flags
from absl import logging

import tensorflow as tf

from tf2_common.training import standard_runnable
from tf2_common.training import utils
from tf2_common.utils.flags import core as flags_core
from tf2_common.utils.mlp_log import mlp_log
import common
import imagenet_preprocessing
import resnet_model

flags.DEFINE_boolean('trace_warmup', default=False,
                     help='Whether or not to programmatically capture an Xprof'
                     ' trace in the warmup loop.')


class _UnwrapPreventer(object):
  """Wrapper that DistributionStrategy will not unwrap.

  Typically, DistributionStrategy will unwrap values when going from a cross-
  replica context to a replica context via `call_for_each_replica`. This class
  is a wrapper that DistributionStrategy will not unwrap, so it can be used to
  prevent it from unwrapping a value.

  TODO(reedwm): Find/implement a better way of preventing values from being
  unwrapped by DistributionStrategy
  """

  __slots__ = ['value']

  def __init__(self, value):
    self.value = value


class ResnetRunnable(standard_runnable.StandardRunnableWithWarmup):
  """Implements the training and evaluation APIs for Resnet model."""

  def __init__(self, flags_obj, time_callback):
    standard_runnable.StandardRunnableWithWarmup.__init__(
        self,
        flags_obj.use_tf_while_loop,
        flags_obj.use_tf_function)

    self.strategy = tf.distribute.get_strategy()
    self.flags_obj = flags_obj
    self.dtype = flags_core.get_tf_dtype(flags_obj)
    self.time_callback = time_callback

    # Input pipeline related
    batch_size = flags_obj.batch_size
    if batch_size % self.strategy.num_replicas_in_sync != 0:
      raise ValueError(
          'Batch size must be divisible by number of replicas : {}'.format(
              self.strategy.num_replicas_in_sync))

    steps_per_epoch, train_epochs = common.get_num_train_iterations(flags_obj)
    if train_epochs > 1:
      train_epochs = flags_obj.train_epochs

    # As auto rebatching is not supported in
    # `experimental_distribute_datasets_from_function()` API, which is
    # required when cloning dataset to multiple workers in eager mode,
    # we use per-replica batch size.
    self.batch_size = int(batch_size / self.strategy.num_replicas_in_sync)

    self.synthetic_input_fn = common.get_synth_input_fn(
        height=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
        width=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
        num_channels=imagenet_preprocessing.NUM_CHANNELS,
        num_classes=self.flags_obj.num_classes,
        dtype=self.dtype,
        drop_remainder=True)

    if self.flags_obj.use_synthetic_data:
      self.input_fn = self.synthetic_input_fn
    else:
      self.input_fn = imagenet_preprocessing.input_fn

    resnet_model.change_keras_layer(flags_obj.use_tf_keras_layers)
    self.model = resnet_model.resnet50(
        num_classes=self.flags_obj.num_classes,
        batch_size=flags_obj.batch_size,
        use_l2_regularizer=not flags_obj.single_l2_loss_op)

    self.use_lars_optimizer = False
    self.num_accumulation_steps = self.flags_obj.num_accumulation_steps
    if self.flags_obj.optimizer == 'LARS':
      self.use_lars_optimizer = True
    self.optimizer, _ = common.get_optimizer(
        flags_obj=flags_obj,
        steps_per_epoch=steps_per_epoch,
        train_steps=steps_per_epoch * train_epochs)
    # Make sure iterations variable is created inside scope.
    self.global_step = self.optimizer.iterations

    if self.dtype == tf.float16:
      loss_scale = flags_core.get_loss_scale(flags_obj, default_for_fp16=128)
      self.optimizer = (
          tf.keras.mixed_precision.experimental.LossScaleOptimizer(
              self.optimizer, loss_scale))
    elif flags_obj.fp16_implementation == 'graph_rewrite':
      # `dtype` is still float32 in this case. We built the graph in float32
      # and let the graph rewrite change parts of it float16.
      if not flags_obj.use_tf_function:
        raise ValueError('--fp16_implementation=graph_rewrite requires '
                         '--use_tf_function to be true')
      loss_scale = flags_core.get_loss_scale(flags_obj, default_for_fp16=128)
      self.optimizer = (
          tf.train.experimental.enable_mixed_precision_graph_rewrite(
              self.optimizer, loss_scale))

    self.one_hot = False
    self.label_smoothing = flags_obj.label_smoothing
    if self.label_smoothing and self.label_smoothing > 0:
      self.one_hot = True

    if flags_obj.report_accuracy_metrics:
      self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
      if self.one_hot:
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(
            'train_accuracy', dtype=tf.float32)
      else:
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            'train_accuracy', dtype=tf.float32)
      self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    else:
      self.train_loss = None
      self.train_accuracy = None
      self.test_loss = None

    if self.one_hot:
      self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(
          'test_accuracy', dtype=tf.float32)
    else:
      self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
          'test_accuracy', dtype=tf.float32)
    # self.test_corrects = tf.keras.metrics.Sum(
    #     'test_corrects', dtype=tf.float32)
    self.num_eval_steps = common.get_num_eval_steps(flags_obj)

    self.checkpoint = tf.train.Checkpoint(
        model=self.model, optimizer=self.optimizer)

    # Handling epochs.
    self.epoch_steps = steps_per_epoch
    self.epoch_helper = utils.EpochHelper(steps_per_epoch, self.global_step)

    self.steps_per_loop = flags_obj.steps_per_loop
    profile_steps = flags_obj.profile_steps
    if profile_steps:
      profile_steps = [int(i) for i in profile_steps.split(',')]
      self.trace_start_step = profile_steps[0] if profile_steps[0] >= 0 else None
      self.trace_end_step = profile_steps[1]
    else:
      self.trace_start_step = None
      self.trace_end_step = None

    self.epochs_between_evals = flags_obj.epochs_between_evals
    self.training_vars = self.model.trainable_variables
    self.accum_grads = []
    self.accum_grads_dtype = tf.float32

    if self.num_accumulation_steps > 1:
      for var in self.training_vars:
        self.accum_grads.append(self.optimizer.add_weight(
            name=var.name + '_accum',
            shape=var.shape,
            dtype=self.accum_grads_dtype,
            initializer='zeros',
            trainable=False,
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.SUM))

  def build_train_dataset(self):
    """See base class."""
    return utils.make_distributed_dataset(
        self.strategy,
        self.input_fn,
        is_training=True,
        data_dir=self.flags_obj.data_dir,
        batch_size=self.batch_size,
        datasets_num_private_threads=self.flags_obj
        .datasets_num_private_threads,
        dtype=self.dtype,
        drop_remainder=self.flags_obj.drop_train_remainder,
        tf_data_experimental_slack=self.flags_obj.tf_data_experimental_slack,
        dataset_cache=self.flags_obj.training_dataset_cache,
        prefetch_batchs=self.flags_obj.training_prefetch_batchs)

  def build_eval_dataset(self):
    """See base class."""
    return utils.make_distributed_dataset(
        self.strategy,
        self.input_fn,
        is_training=False,
        data_dir=self.flags_obj.data_dir,
        batch_size=self.batch_size,
        datasets_num_private_threads=self.flags_obj
        .datasets_num_private_threads,
        dtype=self.dtype,
        drop_remainder=self.flags_obj.drop_eval_remainder,
        tf_data_experimental_slack=self.flags_obj.tf_data_experimental_slack,
        dataset_cache=self.flags_obj.eval_dataset_cache,
        prefetch_batchs=self.flags_obj.eval_prefetch_batchs)

  def build_synthetic_dataset(self):
    """See base class."""
    return utils.make_distributed_dataset(
        self.strategy,
        self.synthetic_input_fn,
        is_training=True,
        data_dir=self.flags_obj.data_dir,
        batch_size=self.batch_size,
        datasets_num_private_threads=self.flags_obj
        .datasets_num_private_threads,
        dtype=self.dtype,
        drop_remainder=self.flags_obj.drop_train_remainder,
        tf_data_experimental_slack=self.flags_obj.tf_data_experimental_slack,
        dataset_cache=self.flags_obj.training_dataset_cache,
        prefetch_batchs=self.flags_obj.training_prefetch_batchs)

  def train_loop_begin(self):
    """See base class."""
    # Reset all metrics
    if self.train_loss:
      self.train_loss.reset_states()
    if self.train_accuracy:
      self.train_accuracy.reset_states()

    self._epoch_begin()
    if self.trace_start_step:
      global_step = self.global_step.numpy()
      next_global_step = global_step + self.steps_per_loop
      if (global_step <= self.trace_start_step and
          self.trace_start_step < next_global_step):
        self.trace_start(global_step)

    self.time_callback.on_batch_begin(self.epoch_helper.batch_index)

  def train_step(self, iterator):
    """See base class."""

    @tf.function(experimental_compile=True)
    def local_step(images, labels):
      """Local computation of a step."""

      with tf.GradientTape() as tape:
        logits = self.model(images, training=True)

        if self.one_hot:
          prediction_loss = tf.keras.losses.categorical_crossentropy(
              labels, logits, label_smoothing=self.label_smoothing)
        else:
          prediction_loss = tf.keras.losses.sparse_categorical_crossentropy(
              labels, logits)
        loss = tf.reduce_sum(prediction_loss) * (
            1.0 / self.flags_obj.batch_size)

        # Save ~3 seconds per epoch on GPU when skipping
        # L2 loss computation; can only skip when using LARS
        # Details in decription of cl/308018913
        if not self.use_lars_optimizer:
          num_replicas = self.strategy.num_replicas_in_sync

          if self.flags_obj.single_l2_loss_op:
            l2_loss = self.flags_obj.weight_decay * 2 * tf.add_n([
                tf.nn.l2_loss(v)
                for v in self.model.trainable_variables
                if 'bn' not in v.name
            ])

            loss += (l2_loss / num_replicas)
          else:
            loss += (tf.reduce_sum(self.model.losses) / num_replicas)

        # Scale the loss
        if self.flags_obj.dtype == 'fp16':
          loss = self.optimizer.get_scaled_loss(loss)

      grads = tape.gradient(loss, self.model.trainable_variables)

      # Unscale the grads
      if self.flags_obj.dtype == 'fp16':
        grads = self.optimizer.get_unscaled_gradients(grads)
      
      return logits, loss, grads

    def _maybe_apply_grads_and_clear(distribution):
      def _apply_grads_and_clear_for_each_replica():
        local_replica_id = tf.get_static_value(
            self.strategy.extended._get_local_replica_id(
                tf.distribute.get_replica_context().replica_id_in_sync_group))
        replica_accum_grads = []
        for accum_grad, var in zip(self.accum_grads, self.training_vars):
          local_accum_grad = self.strategy.experimental_local_results(
              accum_grad)
          replica_accum_grad = local_accum_grad[local_replica_id]
          replica_accum_grad = tf.cast(replica_accum_grad, var.dtype)
          replica_accum_grads.append(replica_accum_grad)

        self.optimizer.apply_gradients(
            zip(replica_accum_grads, self.training_vars))
        for accum_grad in self.accum_grads:
          accum_grad.assign(tf.zeros_like(accum_grad,
                                          dtype=self.accum_grads_dtype),
                            read_value=False)
      def _apply_grads_and_clear():
        distribution.extended.call_for_each_replica(
            _apply_grads_and_clear_for_each_replica,
            args=())
        return self.optimizer.iterations.assign_add(0, read_value=False)

      def _advance_iteration():
        return self.optimizer.iterations.assign_add(1, read_value=False)

      tf.cond(
          tf.equal(self.optimizer.iterations % self.num_accumulation_steps,
                   self.num_accumulation_steps - 1),
          _apply_grads_and_clear,
          _advance_iteration)

    def step_fn(inputs):
      """Function to run on the device."""
      images, labels = inputs
      logits, loss, grads = local_step(images, labels)

      if self.num_accumulation_steps > 1:
        for grad, accum_grad in zip(grads, self.accum_grads):
          accum_grad.assign_add(tf.cast(grad, self.accum_grads_dtype),
                                read_value=False)
        tf.distribute.get_replica_context().merge_call(
            _maybe_apply_grads_and_clear,
            args=())
      else:
        self.optimizer.apply_gradients(zip(grads, self.training_vars))

      if self.train_loss:
        self.train_loss.update_state(loss)
      if self.train_accuracy:
        self.train_accuracy.update_state(labels, logits)

    self.strategy.run(step_fn, args=(next(iterator),))

  def train_loop_end(self):
    """See base class."""
    metrics = {}
    if self.train_loss:
      metrics['train_loss'] = self.train_loss.result()
    if self.train_accuracy:
      metrics['train_accuracy'] = self.train_accuracy.result()

    self.time_callback.on_batch_end(self.epoch_helper.batch_index - 1)

    if self.trace_end_step:
      global_step = self.global_step.numpy()
      next_global_step = global_step + self.steps_per_loop
      if (global_step <= self.trace_end_step and
          self.trace_end_step < next_global_step):
        self.trace_end(global_step)

    self._epoch_end()
    return metrics

  def eval_begin(self):
    """See base class."""
    if self.test_loss:
      self.test_loss.reset_states()
    if self.test_accuracy:
      self.test_accuracy.reset_states()
    # self.test_corrects.reset_states()

    epoch_num = int(self.epoch_helper.current_epoch)
    mlp_log.mlperf_print('eval_start', None,
                         metadata={'epoch_num': epoch_num + 1})

  def eval_step(self, iterator):
    """See base class."""

    def step_fn(inputs):
      """Function to run on the device."""
      images, labels = inputs
      logits = self.model(images, training=False)

      if self.test_loss:
        if self.one_hot:
          loss = tf.keras.losses.categorical_crossentropy(
              labels, logits, label_smoothing=self.label_smoothing)
        else:
          loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
        loss = tf.reduce_sum(loss) * (1.0 / self.flags_obj.batch_size)
        self.test_loss.update_state(loss)

      if self.test_accuracy:
        self.test_accuracy.update_state(labels, logits)
        # tf.print('labels.shape: ', labels.shape,
        #          ', logits.shape: ', logits.shape,
        #          ', result: ', self.test_accuracy.result())
      # self.test_corrects.update_state(
      #     tf.cast(
      #         tf.reduce_sum(
      #             tf.cast(
      #                 tf.equal(
      #                     tf.cast(tf.argmax(logits, axis=1), labels.dtype),
      #                     labels), tf.int32)), tf.float32))

    self.strategy.run(step_fn, args=(next(iterator),))

  def eval_end(self):
    """See base class."""
    epoch_num = int(self.epoch_helper.current_epoch)
    mlp_log.mlperf_print('eval_stop', None,
                         metadata={'epoch_num': epoch_num + 1})

    eval_accuracy = float(self.test_accuracy.result())
    # eval_accuracy = float(self.test_corrects.result()
    #                      ) / imagenet_preprocessing.NUM_IMAGES['validation']
    # eval_accuracy = float(self.test_accuracy.result()) * \
    #     self.flags_obj.batch_size * self.num_eval_steps / \
    #     imagenet_preprocessing.NUM_IMAGES['validation']
    mlp_log.mlperf_print(
        'eval_accuracy', eval_accuracy, metadata={'epoch_num': epoch_num + 1})

    first_epoch_num = max(epoch_num - self.epochs_between_evals + 1, 0)
    epoch_count = self.epochs_between_evals
    if first_epoch_num == 0:
      epoch_count = self.flags_obj.eval_offset_epochs
      if epoch_count == 0:
        epoch_count = self.flags_obj.epochs_between_evals
    mlp_log.mlperf_print(
        'block_stop',
        None,
        metadata={
            'first_epoch_num': first_epoch_num + 1,
            'epoch_count': epoch_count
        })

    continue_training = True
    if eval_accuracy >= self.flags_obj.target_accuracy:
      continue_training = False
    else:
      mlp_log.mlperf_print(
          'block_start',
          None,
          metadata={
              'first_epoch_num': epoch_num + 2,
              'epoch_count': self.epochs_between_evals
          })

    results = {}
    if self.test_loss:
      results['test_loss'] = self.test_loss.result()
    if self.test_accuracy:
      results['test_accuracy'] = self.test_accuracy.result()
    results['continue_training'] = continue_training
    return results

  def warmup_loop_begin(self):
    """See base class."""
    if self.flags_obj.trace_warmup:
      self.trace_start(-3)
    logging.info('Entering the warmup loop.')

  def warmup_loop_end(self):
    """See base class."""
    if self.flags_obj.trace_warmup:
      self.trace_end(-2)
    # Reset the state
    self.model.reset_states()
    tf.keras.backend.set_value(self.optimizer.iterations, 0)
    for accum_grad in self.accum_grads:
      accum_grad.assign(tf.zeros_like(accum_grad,
                                      dtype=self.accum_grads_dtype),
                        read_value=False)
    logging.info('Exiting the warmup loop.')

  def _epoch_begin(self):
    if self.epoch_helper.epoch_begin():
      self.time_callback.on_epoch_begin(self.epoch_helper.current_epoch)

  def _epoch_end(self):
    # mlp_log.mlperf_print('epoch_stop', None)
    if self.epoch_helper.epoch_end():
      self.time_callback.on_epoch_end(self.epoch_helper.current_epoch)

  def trace_start(self, global_step):
    logging.info('Starting tracing at step %d.', global_step)
    tf.profiler.experimental.start(self.flags_obj.model_dir)

  def trace_end(self, global_step):
    logging.info('Ending trace at step %d', global_step)
    tf.profiler.experimental.stop()
