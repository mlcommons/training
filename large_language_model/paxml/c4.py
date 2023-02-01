# coding=utf-8
# Copyright 2022 Google LLC.
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

"""Language Model configurations on the T5/C4 dataset."""

import functools
from typing import List, Optional

from absl import logging
import jax
from jax import numpy as jnp
from paxml import base_experiment
from paxml import experiment_registry
from paxml import seqio_input
from paxml import tasks_lib
from paxml.tasks.lm import model_params
from paxml.tasks.lm.params import lm_cloud
from praxis import base_input
from praxis import base_layer
from praxis import layers
from praxis import optimizers
from praxis import schedules
from praxis.layers import transformers
import seqio
import t5.data
from t5.data import preprocessors as t5_preprocessors


WeightInit = base_layer.WeightInit

GPT_SPM_PATH = (
    'gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model'
)
GPT_EOS_ID = 1
GPT_VOCABULARY = t5.data.SentencePieceVocabulary(GPT_SPM_PATH)
C4_GPT_OUTPUT_FEATURES_LM = {
    'targets': t5.data.Feature(vocabulary=GPT_VOCABULARY, add_eos=False)
}
C4_TRAIN_DATADIR = 'gs://mlperf-llm-public2'
C4_EVAL_DATADIR = 'gs://mlperf-llm-public2'


class TaskRegistry(t5.data.TaskRegistry):
  """Task registry with extra tracking."""

  TASK_NAMES = []

  @classmethod
  def add_versioned_tfds_task(cls,
                              name: str,
                              *,
                              versions: List[str],
                              pinned_version: Optional[str] = None,
                              tfds_name: str,
                              tfds_data_dir: Optional[str] = None,
                              **kwargs) -> List[seqio.Task]:
    tasks = []
    for version in versions:
      tasks.append(
          cls.add(
              f'{name}_{version}',
              seqio.Task,
              source=seqio.TfdsDataSource(
                  tfds_name=f'{tfds_name}:{version}',
                  tfds_data_dir=tfds_data_dir,
              ),
              **kwargs,
          ))
    if pinned_version is not None:
      tasks.append(
          cls.add(
              name,
              seqio.Task,
              source=seqio.TfdsDataSource(
                  tfds_name=f'{tfds_name}:{pinned_version}',
                  tfds_data_dir=tfds_data_dir,
              ),
              **kwargs,
          ))
    return tasks


# C4 corpus for language model pretraining
TaskRegistry.add_versioned_tfds_task(
    'c4_lm_v301_gpt',
    versions=['3.0.4'],
    pinned_version='3.0.4',
    tfds_name='c4/en',
    tfds_data_dir=C4_TRAIN_DATADIR,
    preprocessors=[
        functools.partial(
            t5_preprocessors.rekey,
            key_map={
                'inputs': None,
                'targets': 'text',
            },
        ),
        seqio.preprocessors.tokenize,
        functools.partial(
            t5_preprocessors.reduce_concat_tokens,
            batch_size=4096,
        ),
        t5_preprocessors.split_tokens_to_targets_length,
    ],
    output_features=C4_GPT_OUTPUT_FEATURES_LM,
    metric_fns=[],
    shuffle_buffer_size=10000,
)

TaskRegistry.add_versioned_tfds_task(
    'c4_lm_v301_gpt_eval',
    versions=['3.0.4'],
    pinned_version='3.0.4',
    tfds_name='c4/en',
    tfds_data_dir=C4_EVAL_DATADIR,
    preprocessors=[
        functools.partial(
            t5_preprocessors.rekey,
            key_map={
                'inputs': None,
                'targets': 'text',
            },
        ),
        seqio.preprocessors.tokenize,
        functools.partial(
            t5_preprocessors.reduce_concat_tokens,
            batch_size=24567,
        ),
        t5_preprocessors.split_tokens_to_targets_length,
    ],
    output_features=C4_GPT_OUTPUT_FEATURES_LM,
    metric_fns=[],
    shuffle_buffer_size=None,
)


class C4UnsupervisedDataset(base_experiment.BaseExperiment):
  """Used for training Baseline ULM."""
  PERCORE_BATCH_SIZE = 1
  MAX_SEQ_LEN = 1024

  def _dataset_common(self, is_training) -> base_input.BaseInput.HParams:
    num_local_devices = jax.local_device_count()
    if self.PERCORE_BATCH_SIZE >= 1:
      batch_size_per_process = int(self.PERCORE_BATCH_SIZE * num_local_devices)
      num_infeed_hosts = jax.process_count()
    else:
      global_batch_size = int(self.PERCORE_BATCH_SIZE * num_local_devices *
                              jax.process_count())
      if jax.process_count() > 1:
        assert global_batch_size % num_local_devices == 0
        batch_size_per_process = num_local_devices
        num_infeed_hosts = global_batch_size // batch_size_per_process
      else:
        batch_size_per_process = int(self.PERCORE_BATCH_SIZE *
                                     num_local_devices)
        num_infeed_hosts = 1
    seed = None
    if is_training:
      seed = 9876
      # TODO(sgpyc): enable sync of seeds across hosts, currently the
      # following failed because of "sync_global_devices name mismatch"
      # seed = jnp.int32(multihost_utils.broadcast_one_to_all(seed))
      logging.info('Train input seed: %s',
                   'None' if seed is None else seed)
    p = seqio_input.SeqIOInput.HParams(
        name='C4Train' if is_training else 'C4Validation',
        mixture_name='c4_lm_v301_gpt' if is_training else 'c4_lm_v301_gpt_eval',
        split_name='train2' if is_training else 'validation_24567exp',
        task_feature_lengths={'targets': self.MAX_SEQ_LEN},
        use_cached=False,
        repeat=True if is_training else False,
        feature_converter=seqio_input.LanguageModelFeatures(
            pack=True if is_training else False,
            use_custom_packing_ops=False,
            bos_id=0,
            reverse_bos_padding=True,
            eos_id=GPT_EOS_ID,
        ),
        is_training=is_training,
        input_random_seed=(seed if is_training else 4321),
        batch_size=batch_size_per_process,
        drop_remainder=True if is_training else False,
        num_infeed_hosts=num_infeed_hosts,
        reset_for_eval=False if is_training else True,
        annotate_padding_fields=True,
    )
    return p

  def datasets(self) -> List[base_input.BaseInput.HParams]:
    """Returns a list of dataset parameters."""
    return [
        self._dataset_common(is_training=True),
        self._dataset_common(is_training=False)
    ]


def set_adam_and_learning_rate_schedule(
    cls, task_p: tasks_lib.SingleTask.HParams
) -> tasks_lib.SingleTask.HParams:
  """Sets the Adam optimizer and the learning rate schedule."""
  lp = task_p.train.learner
  lp.loss_name = 'total_loss'
  lp.optimizer = optimizers.Adam.HParams(
      beta1=cls.ADAM_BETA1 if cls.ADAM_BETA1 else 0.9,
      beta2=cls.ADAM_BETA2 if cls.ADAM_BETA2 else 0.999,
      weight_decay=cls.WEIGHT_DECAY if cls.WEIGHT_DECAY else 0.0,
      epsilon=cls.ADAM_EPSILON if cls.ADAM_EPSILON else 1e-6,
      epsilon_root=cls.ADAM_EPSILON_ROOT if cls.ADAM_EPSILON_ROOT else 0.0,
      clip_gradient_norm_to_value=cls.CLIP_GRADIENT_NORM_TO_VALUE
      if cls.CLIP_GRADIENT_NORM_TO_VALUE
      else 5.0,
      clip_threshold=cls.ADAM_CLIP_THRESHOLD
      if cls.ADAM_CLIP_THRESHOLD
      else 1.0,
  )
  lp.optimizer.learning_rate = cls.LEARNING_RATE

  if cls.LR_SCHEDULE == 'linear_rampup_exponential_decay':
    lp.optimizer.lr_schedule = schedules.LinearRampupExponentialDecay.HParams(
        warmup_steps=cls.LR_LRED_WARMUP,
        decay_start=cls.LR_LRED_DECAY_START,
        decay_end=cls.LR_LRED_DECAY_END,
        min_ratio=cls.LR_LRED_MIN_RATIO,
        max=cls.LR_LRED_MAX,
    )
  elif cls.LR_SCHEDULE == 'linear_rampup_cosine_decay':
    lp.optimizer.lr_schedule = schedules.LinearRampupCosineDecay.HParams(
        warmup_steps=cls.LR_COS_WARMUP,
        decay_start=cls.LR_COS_DECAY_START,
        decay_end=cls.LR_COS_DECAY_END,
        min_ratio=cls.LR_COS_MIN_RATIO,
        max=cls.LR_COS_MAX,
    )
  else:
    raise NotImplementedError(
        f'Learning rate schedule {cls.LR_SCHEDULE} is not supported.'
    )

  return task_p


class TransformerLmSpmdAdam(model_params.TransformerLmSpmdAdafactor):
  """Base SPMD Transformer LM configuration using Adam.

  Only things different from TransformerLmSpmdAdafactor are listed.
  """
  # architecture related
  NUM_LAYERS = 32
  NUM_HEADS = 16
  MODEL_DIMS = 1024
  HIDDEN_DIMS = MODEL_DIMS * 4
  FPROP_DTYPE = jnp.float32
  PACKED_INPUT = True
  USE_BIAS = False
  EMBEDDING_LOOKUP_STYLE = 'matmul'

  # optimizer related
  LEARNING_RATE = 1e-3
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.99
  ADAM_CLIP_THRESHOLD = 1.0
  ADAM_EPSILON = 1e-6
  ADAM_EPSILON_ROOT = 0.0

  # Learning rate schedule
  LR_SCHEDULE = 'linear_rampup_exponential_decay'
  LR_LRED_WARMUP = 4000
  LR_LRED_DECAY_START = 4001
  LR_LRED_DECAY_END = 300000
  LR_LRED_MIN_RATIO = 0.1
  LR_LRED_MAX = 1.0

  LR_COS_MIN_RATIO = 0.1
  LR_COS_MAX = 1.0
  LR_COS_WARMUP = 4000
  LR_COS_DECAY_START = 4001
  LR_COS_DECAY_END = 300000

  def task(self) -> tasks_lib.SingleTask.HParams:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model
    model_p.lm_tpl.packed_input = self.PACKED_INPUT  # pytype: disable=attribute-error  # enable-nested-classes

    stacked_p = model_p.lm_tpl.stacked_transformer_tpl  # pytype: disable=attribute-error  # enable-nested-classes
    if stacked_p.cls == transformers.PipelinedTransformer:
      stacked_p = stacked_p.pipeline_stage
    if self.USE_REPEATED_LAYER:
      stacked_p = stacked_p.block
    transformer_layer_p = stacked_p.transformer_layer_params_tpl
    transformer_layer_p.tr_atten_tpl.use_bias = self.USE_BIAS

    task_p = set_adam_and_learning_rate_schedule(cls=self, task_p=task_p)

    return task_p


class TransformerLmSpmdPipelineAdam(
    model_params.TransformerLmSpmdPipelineAdafactor
):
  """Base pipelined SPMD Transformer LM configuration using Adam.

  Only things different from TransformerLmSpmdPipelineAdafactor are listed.
  """

  # architecture related
  NUM_LAYERS = 32
  NUM_HEADS = 16
  MODEL_DIMS = 1024
  HIDDEN_DIMS = MODEL_DIMS * 4
  FPROP_DTYPE = jnp.float32
  PACKED_INPUT = True
  USE_BIAS = False
  EMBEDDING_LOOKUP_STYLE = 'matmul'

  # optimizer related
  LEARNING_RATE = 1e-3
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.99
  ADAM_CLIP_THRESHOLD = 1.0
  ADAM_EPSILON = 1e-6
  ADAM_EPSILON_ROOT = 0.0

  # Learning rate schedule
  LR_SCHEDULE = 'linear_rampup_exponential_decay'
  LR_LRED_WARMUP = 4000
  LR_LRED_DECAY_START = 4001
  LR_LRED_DECAY_END = 300000
  LR_LRED_MIN_RATIO = 0.1
  LR_LRED_MAX = 1.0

  LR_COS_MIN_RATIO = 0.1
  LR_COS_MAX = 1.0
  LR_COS_WARMUP = 4000
  LR_COS_DECAY_START = 4001
  LR_COS_DECAY_END = 300000

  def task(self) -> tasks_lib.SingleTask.HParams:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model
    model_p.lm_tpl.packed_input = self.PACKED_INPUT  # pytype: disable=attribute-error  # enable-nested-classes

    stacked_p = model_p.lm_tpl.stacked_transformer_tpl  # pytype: disable=attribute-error  # enable-nested-classes
    if stacked_p.cls == transformers.PipelinedTransformer:
      stacked_p = stacked_p.pipeline_stage
    if self.USE_REPEATED_LAYER:
      stacked_p = stacked_p.block
    transformer_layer_p = stacked_p.transformer_layer_params_tpl
    transformer_layer_p.tr_atten_tpl.use_bias = self.USE_BIAS

    task_p = set_adam_and_learning_rate_schedule(cls=self, task_p=task_p)

    return task_p


@experiment_registry.register
class LmCloudSpmdAdam(TransformerLmSpmdAdam, lm_cloud.SyntheticDataset):
  """Base config for an SPMD model."""

  NUM_LAYERS = 2
  MODEL_DIMS = 2048
  HIDDEN_DIMS = MODEL_DIMS * 4
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False

  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # Sub-class has to specify a mesh.
  ICI_MESH_SHAPE = [1, 4, 2]


def configure_gpt3_task(
    cls,
    task_p: tasks_lib.SingleTask.HParams,
) -> tasks_lib.SingleTask.HParams:
  """Returns task with gpt3 related configs."""
  model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes

  model_p.decoder_tpl.eos_id = (
      GPT_EOS_ID  # pytype: disable=attribute-error  # enable-nested-classes
  )
  model_p.decoder_tpl.seqlen = cls.MAX_SEQ_LEN  # pytype: disable=attribute-error  # enable-nested-classes

  model_p.params_init = WeightInit.Gaussian(0.006)

  softmax_init = WeightInit.Gaussian(0.006)
  model_p.lm_tpl.softmax_tpl.params_init = softmax_init
  model_p.lm_tpl.softmax_tpl.feed_forward_tpl.has_bias = False
  model_p.lm_tpl.softmax_tpl.soft_cap_logits = None

  if cls.SEPARATE_EMBEDDING:
    model_p.lm_tpl.separate_embedding_tpl.scale_sqrt_depth = False
    model_p.lm_tpl.separate_embedding_tpl.lookup_style = (
        cls.EMBEDDING_LOOKUP_STYLE
    )
  else:
    model_p.lm_tpl.softmax_tpl.scale_sqrt_depth = False
    model_p.lm_tpl.softmax_tpl.lookup_style = cls.EMBEDDING_LOOKUP_STYLE
  if cls.TRAINABLE_POSITION_EMB:
    model_p.lm_tpl.position_emb_tpl.lookup_style = cls.EMBEDDING_LOOKUP_STYLE

  stacked_p = model_p.lm_tpl.stacked_transformer_tpl
  if stacked_p.cls == transformers.PipelinedTransformer:
    stacked_p = stacked_p.pipeline_stage
  if issubclass(stacked_p.cls, transformers.StackedTransformerRepeated):
    stacked_p = stacked_p.block
  transformer_layer_p = stacked_p.transformer_layer_params_tpl

  transformer_layer_p.ln_tpl.epsilon = cls.LAYERNORM_EPSILON
  transformer_layer_p.tr_fflayer_tpl.ln_tpl.epsilon = cls.LAYERNORM_EPSILON
  model_p.lm_tpl.final_ln_tpl.epsilon = cls.LAYERNORM_EPSILON
  transformer_layer_p.tr_atten_tpl.internal_enable_per_dim_scale = False
  transformer_layer_p.tr_atten_tpl.use_bias = True

  transformer_layer_p.tr_fflayer_tpl.activation_tpl.approximate = True

  for atten_p in (
      transformer_layer_p.tr_atten_tpl,
      transformer_layer_p.cross_atten_tpl,
  ):
    if atten_p is None:
      continue
    atten_wp = atten_p.weight_split_dims_mapping
    atten_wp.proj = ['data', 'mdl', None]

  return task_p


@experiment_registry.register
class C4SpmdAdam(TransformerLmSpmdAdam,
                 C4UnsupervisedDataset):
  r"""Base config for a decoder only transformer."""

  NUM_LAYERS = 24
  NUM_HEADS = 32
  MODEL_DIMS = 2048
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  # Defaults to MODEL_DIMS // NUM_HEADS.
  DIMS_PER_HEAD = None
  # Known as NUM_EMBEDDINGS in t5x
  VOCAB_SIZE = 32128
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_FOR_MLPERF_200B
  CHECKPOINT_EVERY_N_STEPS = 1000

  # Sub-class has to specify a mesh.
  ICI_MESH_SHAPE = [1, 4, 2]

  def task(self) -> tasks_lib.SingleTask.HParams:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes
    model_p.decoder_tpl.eos_id = GPT_EOS_ID  # pytype: disable=attribute-error  # enable-nested-classes
    model_p.decoder_tpl.seqlen = self.MAX_SEQ_LEN  # pytype: disable=attribute-error  # enable-nested-classes

    return task_p


class C4SpmdGpt3AdamOrgHP(C4SpmdAdam):
  r"""GPT-3 config with original HPs.

  From the paper & after convergence matching with
  NVIDIA's Megatron-LM framework.
  """
  MAX_SEQ_LEN = 2048

  NUM_LAYERS = 96
  NUM_HEADS = 96
  MODEL_DIMS = 12288
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  # Defaults to MODEL_DIMS // NUM_HEADS.
  DIMS_PER_HEAD = None
  # Known as NUM_EMBEDDINGS in t5x
  VOCAB_SIZE = 50257
  USE_REPEATED_LAYER = True

  # HPs
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False
  SEPARATE_EMBEDDING = False
  TRAINABLE_POSITION_EMB = True
  TRAINABLE_PE_MAX_SEQ_LEN = 16384
  ATTEN_LOGIT_CAP = -1.0  # Disable logits cap in atten

  LEARNING_RATE = 6e-5
  WEIGHT_DECAY = 0.1
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.95
  ADAM_EPSILON = 1e-8
  ADAM_CLIP_THRESHOLD = -1.0  # Disable Adam clip_threshold
  CLIP_GRADIENT_NORM_TO_VALUE = 1.0
  LAYERNORM_EPSILON = 1e-5

  # In units of steps for BS1.5k
  LR_SCHEDULE = 'linear_rampup_cosine_decay'
  LR_COS_WARMUP = 265
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 108600
  LR_COS_MAX = 1.0
  LR_COS_MIN_RATIO = 0.1

  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # Checkpoint
  EVAL_INTERVAL_STEPS = 100
  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_EVERY_N_STEPS = 100
  CHECKPOINT_MAX_TO_KEEP = 10

  def task(self) -> tasks_lib.SingleTask.HParams:
    """Returns the task parameters."""
    task_p = super().task()
    task_p = configure_gpt3_task(self, task_p)
    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamOrgHPBS1p5k1536Replicas(C4SpmdGpt3AdamOrgHP):
  r"""GPT-3 config in fp32 for 1536 replicas with 1536 global batch size."""
  # Padded to TPU friendly size
  VOCAB_SIZE = 51200

  PERCORE_BATCH_SIZE = 1
  ICI_MESH_SHAPE = [1, 64, 24]
  FPROP_DTYPE = jnp.float32
  CHECKPOINT_MAX_TO_KEEP = 100
  EVAL_INTERVAL_STEPS = 25
  SUMMARY_INTERVAL_STEPS = 1


@experiment_registry.register
class C4SpmdPipelineAdam(TransformerLmSpmdPipelineAdam, C4UnsupervisedDataset):
  r"""Base config for a decoder only transformer with pipeline."""

  NUM_LAYERS = 24
  NUM_HEADS = 32
  MODEL_DIMS = 2048
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  # Defaults to MODEL_DIMS // NUM_HEADS.
  DIMS_PER_HEAD = None
  # Known as NUM_EMBEDDINGS in t5x
  VOCAB_SIZE = 32128
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_FOR_MLPERF_200B
  CHECKPOINT_EVERY_N_STEPS = 1000

  # Sub-class has to specify a mesh.
  MICROBATCH_SIZE = 2
  ICI_MESH_SHAPE = [2, 1, 2, 2]
  NUM_STAGES = 2
  EMB_W_DATA_DIMS = ('replica', 'data')

  def task(self) -> tasks_lib.SingleTask.HParams:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes
    model_p.decoder_tpl.eos_id = (
        GPT_EOS_ID  # pytype: disable=attribute-error  # enable-nested-classes
    )
    model_p.decoder_tpl.seqlen = self.MAX_SEQ_LEN  # pytype: disable=attribute-error  # enable-nested-classes

    return task_p


class C4SpmdPipelineGpt3AdamOrgHP(C4SpmdPipelineAdam):
  r"""GPT-3 config with original HPs.

  From the paper & after convergence matching with
  NVIDIA's Megatron-LM framework.
  """
  MAX_SEQ_LEN = 2048

  NUM_LAYERS = 96
  NUM_HEADS = 96
  MODEL_DIMS = 12288
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  # Defaults to MODEL_DIMS // NUM_HEADS.
  DIMS_PER_HEAD = None
  # Known as NUM_EMBEDDINGS in t5x
  VOCAB_SIZE = 50257
  USE_REPEATED_LAYER = False

  # HPs
  ACTIVATION_CLS = layers.GELU
  USE_GATED_ACTIVATION = False
  SEPARATE_EMBEDDING = False
  TRAINABLE_POSITION_EMB = True
  TRAINABLE_PE_MAX_SEQ_LEN = 16384
  ATTEN_LOGIT_CAP = -1.0  # Disable logits cap in atten

  LEARNING_RATE = 6e-5
  WEIGHT_DECAY = 0.1
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.95
  ADAM_EPSILON = 1e-8
  ADAM_CLIP_THRESHOLD = -1.0  # Disable Adam clip_threshold
  CLIP_GRADIENT_NORM_TO_VALUE = 1.0
  LAYERNORM_EPSILON = 1e-5

  # In units of steps for BS1.5k
  LR_SCHEDULE = 'linear_rampup_cosine_decay'
  LR_COS_WARMUP = 265
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 108600
  LR_COS_MAX = 1.0
  LR_COS_MIN_RATIO = 0.1

  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # Checkpoint
  EVAL_INTERVAL_STEPS = 100
  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_EVERY_N_STEPS = 100
  CHECKPOINT_MAX_TO_KEEP = 10

  def task(self) -> tasks_lib.SingleTask.HParams:
    """Returns the task parameters."""
    task_p = super().task()
    task_p = configure_gpt3_task(self, task_p)
    return task_p


@experiment_registry.register
class C4SpmdPipelineGpt3AdamOrgHPBS1p5k768Replicas(C4SpmdPipelineGpt3AdamOrgHP):
  r"""GPT-3 config in fp32 for 768 replicas with 1536 global batch size."""
  # Padded to TPU friendly size
  VOCAB_SIZE = 51200

  PERCORE_BATCH_SIZE = 2
  NUM_STAGES = 8
  ICI_MESH_SHAPE = [8, 1, 8, 12]
  MICROBATCH_SIAZE = 8
  FPROP_DTYPE = jnp.float32
  CHECKPOINT_MAX_TO_KEEP = 100
  EVAL_INTERVAL_STEPS = 25
  SUMMARY_INTERVAL_STEPS = 1
  CHECKPOINT_EVERY_N_STEPS = 50
  STREAM_IO = False


@experiment_registry.register
class C4SpmdPipelineGpt3AdamMLPerfHPBS1p5k768Replicas(
    C4SpmdPipelineGpt3AdamOrgHP
):
  r"""GPT-3 config in fp32 for 768 replicas with 1536 global batch size."""
  VOCAB_SIZE = 51200
  PERCORE_BATCH_SIZE = 2
  NUM_STAGES = 8
  ICI_MESH_SHAPE = [8, 1, 8, 12]
  # NUM_MICROBATCHS = 192
  MICROBATCH_SIZE = 8
  FPROP_DTYPE = jnp.float32
  CHECKPOINT_MAX_TO_KEEP = 100
  EVAL_INTERVAL_STEPS = 16
  SUMMARY_INTERVAL_STEPS = 1
  CHECKPOINT_EVERY_N_STEPS = 32

  LEARNING_RATE = 2e-5
  STREAM_IO = False


@experiment_registry.register
class C4SpmdPipelineGpt3AdamMLPerfHPBS2k512Replicas(
    C4SpmdPipelineGpt3AdamOrgHP
):
  r"""GPT-3 config in fp32 for 512 replicas with 2k global batch size."""
  VOCAB_SIZE = 51200
  PERCORE_BATCH_SIZE = 4
  NUM_STAGES = 8
  ICI_MESH_SHAPE = [8, 1, 8, 8]
  # NUM_MICROBATCHS = 256
  MICROBATCH_SIZE = 8
  # FPROP_DTYPE = jnp.bfloat16
  FPROP_DTYPE = jnp.float32
  CHECKPOINT_MAX_TO_KEEP = 100
  EVAL_INTERVAL_STEPS = 12
  SUMMARY_INTERVAL_STEPS = 1
  CHECKPOINT_EVERY_N_STEPS = 24

  LEARNING_RATE = 2e-5
  STREAM_IO = True
  LR_COS_WARMUP = 199
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 81450


@experiment_registry.register
class C4SpmdPipelineGpt3AdamMLPerfHPBS3k768Replicas(
    C4SpmdPipelineGpt3AdamOrgHP
):
  r"""GPT-3 config in fp32 for 768 replicas with 3072 global batch size."""
  VOCAB_SIZE = 51200
  PERCORE_BATCH_SIZE = 4
  NUM_STAGES = 4
  ICI_MESH_SHAPE = [4, 1, 16, 12]
  # NUM_MICROBATCHS = 192
  MICROBATCH_SIZE = 16
  FPROP_DTYPE = jnp.float32
  CHECKPOINT_MAX_TO_KEEP = 100
  EVAL_INTERVAL_STEPS = 8
  SUMMARY_INTERVAL_STEPS = 1
  CHECKPOINT_EVERY_N_STEPS = 16

  LEARNING_RATE = 2e-5
  STREAM_IO = True
  LR_COS_WARMUP = 133
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 54300


@experiment_registry.register
class C4SpmdPipelineGpt3AdamMLPerfHPBS4k1024Replicas(
    C4SpmdPipelineGpt3AdamOrgHP
):
  r"""GPT-3 config in fp32 for 1024 replicas with 4096 global batch size."""
  VOCAB_SIZE = 51200
  PERCORE_BATCH_SIZE = 4
  NUM_STAGES = 8
  ICI_MESH_SHAPE = [8, 1, 8, 16]
  # NUM_MICROBATCHS = 512
  MICROBATCH_SIZE = 8
  FPROP_DTYPE = jnp.float32
  CHECKPOINT_MAX_TO_KEEP = 36
  EVAL_INTERVAL_STEPS = 6
  SUMMARY_INTERVAL_STEPS = 1
  CHECKPOINT_EVERY_N_STEPS = 12

  LEARNING_RATE = 3e-5
  STREAM_IO = True
  LR_COS_WARMUP = 99
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 40725


@experiment_registry.register
class C4SpmdPipelineGpt3AdamMLPerfHPBS8k1024Replicas(
    C4SpmdPipelineGpt3AdamOrgHP
):
  r"""GPT-3 config in fp32 for 1024 replicas with 8192 global batch size."""
  VOCAB_SIZE = 51200
  PERCORE_BATCH_SIZE = 8
  NUM_STAGES = 4
  ICI_MESH_SHAPE = [4, 1, 16, 16]
  # NUM_MICROBATCHS = 512
  MICROBATCH_SIZE = 16
  FPROP_DTYPE = jnp.float32
  CHECKPOINT_MAX_TO_KEEP = 36
  EVAL_INTERVAL_STEPS = 3
  SUMMARY_INTERVAL_STEPS = 1
  CHECKPOINT_EVERY_N_STEPS = 6

  LEARNING_RATE = 3e-5
  STREAM_IO = True
  LR_COS_WARMUP = 50
  LR_COS_DECAY_START = LR_COS_WARMUP + 1
  LR_COS_DECAY_END = 20363


@experiment_registry.register
class C4Spmd1BAdam4Replicas(C4SpmdAdam):
  r"""GPT-3 config with 1B params.

  Model Parameters:  Global batch size = 1 * 4 * 1 * 32 = 128
  """
  NUM_LAYERS = 13
  MODEL_DIMS = 2560
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 20
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 32
  MAX_SEQ_LEN = 1024
  VOCAB_SIZE = 32000
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 4, 1]


@experiment_registry.register
class C4Spmd2BAdam4Replicas(C4SpmdAdam):
  r"""GPT-3 config with 2B params.

  Model Parameters: Global batch size = 1 * 4 * 1 * 32 = 128.
  """
  NUM_LAYERS = 18
  MODEL_DIMS = 3072
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 24
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 32
  MAX_SEQ_LEN = 1024
  VOCAB_SIZE = 32000
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 4, 1]

  def task(self) -> tasks_lib.SingleTask.HParams:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.train.learner.repeat_prefix_sep = '_'
    return task_p


@experiment_registry.register
class C4Spmd16BAdam32Replicas(C4SpmdAdam):
  r"""GPT-3 config with 16B params.

  Model Parameters: Global batch size = 1 * 2 * 16 * 16 = 512.
  """
  NUM_LAYERS = 36
  MODEL_DIMS = 6144
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 48
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 16
  MAX_SEQ_LEN = 1024
  VOCAB_SIZE = 32000
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 16, 2]

  def task(self) -> tasks_lib.SingleTask.HParams:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.train.learner.repeat_prefix_sep = '_'
    return task_p


@experiment_registry.register
class C4Spmd32BAdam64Replicas(C4SpmdAdam):
  r"""GPT-3 config with 32B params.

  Model Parameters: Global batch size = 1 * 16 * 4 * 8 = 512.
  """
  NUM_LAYERS = 40
  MODEL_DIMS = 8192
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 64
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 8
  MAX_SEQ_LEN = 1024
  VOCAB_SIZE = 32000
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 16, 4]

  def task(self) -> tasks_lib.SingleTask.HParams:
    """Returns the task parameters."""
    task_p = super().task()
    task_p.train.learner.repeat_prefix_sep = '_'
    return task_p


@experiment_registry.register
class C4SpmdGpt3L16AdamOrgHP(C4SpmdGpt3AdamOrgHP):
  r"""Small GPT-3 config in bf16 for 64 replicas with 192 global batch size."""
  NUM_LAYERS = 16
  FPROP_DTYPE = jnp.bfloat16
  PERCORE_BATCH_SIZE = 3
  EVAL_INTERVAL_STEPS = 25000
  ICI_MESH_SHAPE = [1, 16, 4]


@experiment_registry.register
class C4SpmdPipelineGpt3SmallAdam64Replicas(C4SpmdPipelineGpt3AdamOrgHP):
  """Small GPT-3 config in bf16 for 64 replicas with 512 global batch size.

  This was called GPT-3 XL in the GPT-3 paper, with 1.3B parameters.
  """
  NUM_STAGES = 4
  NUM_LAYERS = 24
  NUM_HEADS = 24
  MODEL_DIMS = 3072
  # Known as MLP_DIM in t5x
  HIDDEN_DIMS = MODEL_DIMS * 4
  DIMS_PER_HEAD = 128

  PER_CORE_BATCH_SIZE = 8
  MICROBATCH_SIZE = 16
  FPROP_DTYPE = jnp.bfloat16
  LEARNING_RATE = 2.0e-4
  ICI_MESH_SHAPE = [4, 1, 4, 4]

  CHECKPOINT_MAX_TO_KEEP = 1000
  EVAL_INTERVAL_STEPS = 100
  SUMMARY_INTERVAL_STEPS = 1
  CHECKPOINT_EVERY_N_STEPS = 200
