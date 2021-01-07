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
"""Run masked LM/next sentence masked_lm pre-training for BERT in TF 2.x."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import optimization
import bert_models
import common_flags
import configs
import input_pipeline
from tf2_common.modeling import model_training_utils
from tf2_common.modeling import performance
from tf2_common.utils.misc import distribution_utils
from tf2_common.utils.mlp_log import mlp_log


flags.DEFINE_string('train_files', None,
                    'File path to retrieve training data for pre-training.')
flags.DEFINE_string('eval_files', None,
                    'File path to retrieve eval data for pre-training.')
# Model training specific flags.
flags.DEFINE_integer(
    'max_seq_length', 128,
    'The maximum total input sequence length after WordPiece tokenization. '
    'Sequences longer than this will be truncated, and sequences shorter '
    'than this will be padded.')
flags.DEFINE_integer('max_predictions_per_seq', 20,
                     'Maximum predictions per sequence_output.')
flags.DEFINE_integer('train_batch_size', 32, 'Total batch size for training.')
flags.DEFINE_integer('num_steps_per_epoch', 1000,
                     'Total number of training steps to run per epoch.')
flags.DEFINE_float('warmup_steps', 10000,
                   'Warmup steps for optimizer.')
flags.DEFINE_integer('start_warmup_step', 0,
                     'The starting step of warmup.')
flags.DEFINE_integer('stop_steps', None,
                     'The number of steps to stop training.')
flags.DEFINE_bool('do_eval', False, 'Whether to run eval.')
flags.DEFINE_bool('device_warmup', False,
                  'Whether or not to enable device warmup.')
flags.DEFINE_integer('steps_between_eval', 10000,
                     'Steps between an eval. Is multiple of steps per loop.')
flags.DEFINE_integer('steps_before_eval_start', 0,
                     'Steps before starting eval.')
flags.DEFINE_integer('num_eval_samples', 10000, 'Number of eval samples.')
flags.DEFINE_integer('eval_batch_size', 32, 'Total batch size for training.')
flags.DEFINE_float('weight_decay_rate', 0.01,
                   'The weight_decay_rate value for the optimizer.')
flags.DEFINE_float('beta_1', 0.9, 'The beta_1 value for the optimizer.')
flags.DEFINE_float('beta_2', 0.999, 'The beta_2 value for the optimizer.')
flags.DEFINE_float('epsilon', 1e-6, 'The epsilon value for the optimizer.')
flags.DEFINE_integer('num_accumulation_steps', 1,
                     'number of steps to accumulate with large batch size.')
flags.DEFINE_float('stop_threshold', 0.712, 'Stop threshold for MLPerf.')
flags.DEFINE_float('poly_power', 1.0, 'The power of poly decay.')
common_flags.define_common_bert_flags()

FLAGS = flags.FLAGS


def get_pretrain_dataset_fn(input_file_pattern,
                            seq_length,
                            max_predictions_per_seq,
                            global_batch_size,
                            is_training,
                            use_synthetic,
                            num_eval_samples):
  """Returns input dataset from input file string."""
  def _dataset_fn(ctx=None):
    """Returns tf.data.Dataset for distributed BERT pretraining."""
    if use_synthetic:
      input_patterns = ''
    else:
      input_patterns = input_file_pattern.split(',')
    batch_size = ctx.get_per_replica_batch_size(global_batch_size)

    dataset = input_pipeline.create_pretrain_dataset(
        input_patterns=input_patterns,
        seq_length=seq_length,
        max_predictions_per_seq=max_predictions_per_seq,
        batch_size=batch_size,
        is_training=is_training,
        use_synthetic=use_synthetic,
        input_pipeline_context=ctx,
        num_eval_samples=num_eval_samples)
    return dataset

  return _dataset_fn


def get_loss_fn(loss_factor=1.0):
  """Returns loss function for BERT pretraining."""

  def _bert_pretrain_loss_fn(unused_labels, losses, **unused_args):
    return tf.reduce_mean(losses) * loss_factor

  return _bert_pretrain_loss_fn


def clip_by_global_norm(grads_and_vars):
  grads, tvars = list(zip(*grads_and_vars))
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
  return zip(grads, tvars)


def run_customized_training(strategy,
                            optimizer_type,
                            weight_decay_rate,
                            beta_1,
                            beta_2,
                            epsilon,
                            bert_config,
                            max_seq_length,
                            max_predictions_per_seq,
                            model_dir,
                            steps_per_epoch,
                            steps_per_loop,
                            epochs,
                            initial_lr,
                            warmup_steps,
                            train_files,
                            train_batch_size,
                            do_eval,
                            eval_files,
                            eval_batch_size,
                            num_eval_samples,
                            custom_callbacks,
                            init_checkpoint,
                            steps_between_eval,
                            steps_before_eval_start,
                            stop_threshold,
                            explicit_allreduce,
                            allreduce_bytes_per_pack,
                            enable_checkpoint_and_summary,
                            device_warmup):
  """Run BERT pretrain model training using low-level API."""
  mlp_log.mlperf_print('cache_clear', True)
  mlp_log.mlperf_print('init_start', None)
  mlp_log.mlperf_print('global_batch_size', train_batch_size)
  mlp_log.mlperf_print('max_sequence_length', max_seq_length)
  mlp_log.mlperf_print('max_predictions_per_seq', max_predictions_per_seq)
  mlp_log.mlperf_print('opt_base_learning_rate', initial_lr)
  mlp_log.mlperf_print('opt_lamb_weight_decay_rate', weight_decay_rate)
  mlp_log.mlperf_print('opt_lamb_beta_1', beta_1)
  mlp_log.mlperf_print('opt_lamb_beta_2', beta_2)
  mlp_log.mlperf_print('opt_gradient_accumulation_steps',
                       FLAGS.num_accumulation_steps)
  mlp_log.mlperf_print('opt_learning_rate_warmup_epochs',
                       train_batch_size * warmup_steps)
  mlp_log.mlperf_print('opt_learning_rate_warmup_steps', warmup_steps)
  mlp_log.mlperf_print('num_warmup_steps', warmup_steps)
  mlp_log.mlperf_print('start_warmup_step', FLAGS.start_warmup_step)
  mlp_log.mlperf_print('opt_epsilon', epsilon)
  mlp_log.mlperf_print('eval_samples', num_eval_samples)
  mlp_log.mlperf_print('opt_lamb_learning_rate_decay_poly_power',
                       FLAGS.poly_power)
  mlp_log.mlperf_print('opt_learning_rate_training_steps',
                       steps_per_epoch * epochs)
  mlp_log.mlperf_print('train_samples',
                       train_batch_size * steps_per_epoch * epochs)
  train_input_fn = get_pretrain_dataset_fn(
      input_file_pattern=train_files,
      seq_length=max_seq_length,
      max_predictions_per_seq=max_predictions_per_seq,
      global_batch_size=train_batch_size,
      is_training=True,
      num_eval_samples=num_eval_samples,
      use_synthetic=False)
  eval_input_fn = None
  if do_eval:
    eval_input_fn = get_pretrain_dataset_fn(
        input_file_pattern=eval_files,
        seq_length=max_seq_length,
        max_predictions_per_seq=max_predictions_per_seq,
        global_batch_size=eval_batch_size,
        is_training=False,
        num_eval_samples=num_eval_samples,
        use_synthetic=False)
  synthetic_train_input_fn = None
  if device_warmup:
    synthetic_train_input_fn = get_pretrain_dataset_fn(
        input_file_pattern=None,
        seq_length=max_seq_length,
        max_predictions_per_seq=max_predictions_per_seq,
        global_batch_size=train_batch_size,
        is_training=True,
        num_eval_samples=1,
        use_synthetic=True)

  def _get_pretrain_model():
    """Gets a pretraining model."""
    pretrain_model, core_model, core_pretrain_model = bert_models.pretrain_model(
        bert_config, max_seq_length, max_predictions_per_seq)
    optimizer = optimization.create_optimizer(
        initial_lr, steps_per_epoch * epochs, warmup_steps,
        optimizer_type=optimizer_type, poly_power=FLAGS.poly_power,
        start_warmup_step=FLAGS.start_warmup_step,
        weight_decay_rate=weight_decay_rate,
        beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
    pretrain_model.optimizer = performance.configure_optimizer(
        optimizer,
        use_float16=common_flags.use_float16(),
        use_graph_rewrite=common_flags.use_graph_rewrite())
    return pretrain_model, core_model, core_pretrain_model

  trained_model, masked_lm_accuracy, run_steps = model_training_utils.run_customized_training_loop(
      strategy=strategy,
      model_fn=_get_pretrain_model,
      loss_fn=get_loss_fn(
          loss_factor=1.0 /
          strategy.num_replicas_in_sync if FLAGS.scale_loss else 1.0),
      model_dir=model_dir,
      train_input_fn=train_input_fn,
      steps_per_epoch=steps_per_epoch,
      steps_per_loop=steps_per_loop,
      epochs=epochs,
      eval_input_fn=eval_input_fn,
      eval_steps=math.ceil(num_eval_samples / eval_batch_size),
      steps_between_eval=steps_between_eval,
      steps_before_eval_start=steps_before_eval_start,
      sub_model_export_name='pretrained/bert_model',
      init_checkpoint=init_checkpoint,
      custom_callbacks=custom_callbacks,
      device_warmup=device_warmup,
      synthetic_train_input_fn=synthetic_train_input_fn,
      explicit_allreduce=explicit_allreduce,
      post_allreduce_callbacks=[clip_by_global_norm],
      allreduce_bytes_per_pack=allreduce_bytes_per_pack,
      enable_checkpoint_and_summary=enable_checkpoint_and_summary,
      num_accumulation_steps=FLAGS.num_accumulation_steps,
      stop_steps=FLAGS.stop_steps,
      stop_threshold=stop_threshold)

  return trained_model, masked_lm_accuracy, run_steps


def run_bert_pretrain(strategy, custom_callbacks=None):
  """Runs BERT pre-training."""
  bert_config = configs.BertConfig.from_json_file(FLAGS.bert_config_file)
  if not strategy:
    raise ValueError('Distribution strategy is not specified.')

  # Runs customized training loop.
  logging.info('Training using customized training loop TF 2.0 with distrubuted'
               'strategy.')

  performance.set_mixed_precision_policy(common_flags.dtype())

  _, masked_lm_accuracy, run_steps = run_customized_training(
      strategy=strategy,
      optimizer_type=FLAGS.optimizer_type,
      weight_decay_rate=FLAGS.weight_decay_rate,
      beta_1=FLAGS.beta_1,
      beta_2=FLAGS.beta_2,
      epsilon=FLAGS.epsilon,
      bert_config=bert_config,
      max_seq_length=FLAGS.max_seq_length,
      max_predictions_per_seq=FLAGS.max_predictions_per_seq,
      model_dir=FLAGS.model_dir,
      steps_per_epoch=FLAGS.num_steps_per_epoch,
      steps_per_loop=FLAGS.steps_per_loop,
      epochs=FLAGS.num_train_epochs,
      initial_lr=FLAGS.learning_rate,
      warmup_steps=FLAGS.warmup_steps,
      train_files=FLAGS.train_files,
      train_batch_size=FLAGS.train_batch_size,
      eval_files=FLAGS.eval_files,
      eval_batch_size=FLAGS.eval_batch_size,
      do_eval=FLAGS.do_eval,
      num_eval_samples=FLAGS.num_eval_samples,
      steps_between_eval=FLAGS.steps_between_eval,
      steps_before_eval_start=FLAGS.steps_before_eval_start,
      stop_threshold=FLAGS.stop_threshold,
      explicit_allreduce=FLAGS.explicit_allreduce,
      allreduce_bytes_per_pack=FLAGS.allreduce_bytes_per_pack,
      enable_checkpoint_and_summary=FLAGS.enable_checkpoint_and_summary,
      custom_callbacks=custom_callbacks,
      init_checkpoint=FLAGS.init_checkpoint,
      device_warmup=FLAGS.device_warmup)
  return masked_lm_accuracy, run_steps


def main(_):
  # Users should always run this script under TF 2.x
  tf.compat.v2.enable_v2_behavior()

  if not FLAGS.model_dir:
    FLAGS.model_dir = '/tmp/bert20/'
  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.distribution_strategy,
      num_gpus=FLAGS.num_gpus,
      all_reduce_alg=FLAGS.all_reduce_alg,
      tpu_address=FLAGS.tpu)
  if strategy:
    print('***** Number of cores used : ', strategy.num_replicas_in_sync)

  run_bert_pretrain(strategy)


if __name__ == '__main__':
  app.run(main)
