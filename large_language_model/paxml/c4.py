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
import math
from typing import List, Optional

import jax
from jax import numpy as jnp
from paxml import base_experiment
from paxml import base_task
from paxml import experiment_registry
from paxml import seqio_input
from paxml.tasks.lm import model_params
from paxml.tasks.lm.params import lm_cloud
from praxis import base_input
from praxis import base_layer
from praxis import layers
from praxis import optimizers
from praxis import schedules
import seqio
import t5.data
from t5.data import preprocessors as t5_preprocessors

WeightInit = base_layer.WeightInit

GPT_SPM_PATH = 'gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model'
GPT_VOCABULARY = t5.data.SentencePieceVocabulary(GPT_SPM_PATH)
C4_GPT_OUTPUT_FEATURES_LM = {
    'targets': t5.data.Feature(vocabulary=GPT_VOCABULARY, add_eos=True)
}
C4_TFDS_DATADIR = 'gs://mlperf-llm-public2'


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
    versions=['3.0.1'],
    pinned_version='3.0.1',
    tfds_name='c4/en',
    tfds_data_dir=C4_TFDS_DATADIR,
    preprocessors=[
        functools.partial(
            t5_preprocessors.rekey,
            key_map={
                'inputs': None,
                'targets': 'text',
            }),
        seqio.preprocessors.tokenize,
        t5_preprocessors.reduce_concat_tokens,
        t5_preprocessors.split_tokens_to_targets_length,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=C4_GPT_OUTPUT_FEATURES_LM,
    metric_fns=[],
    shuffle_buffer_size=100000,
)


class C4UnsupervisedDataset(base_experiment.BaseExperiment):
  """Used for training Baseline ULM."""
  PERCORE_BATCH_SIZE = 1
  MAX_SEQ_LEN = 1024

  def _dataset_common(self, is_training) -> base_input.BaseInput.HParams:
    num_local_devices = jax.local_device_count()
    batch_size = self.PERCORE_BATCH_SIZE * num_local_devices
    p = seqio_input.SeqIOInput.HParams(
        name='C4Train' if is_training else 'C4Validation',
        mixture_name='c4_lm_v301_gpt',
        split_name='train' if is_training else 'validation',
        task_feature_lengths={'targets': self.MAX_SEQ_LEN},
        use_cached=True,
        repeat=True if is_training else False,
        feature_converter=seqio_input.LanguageModelFeatures(
            pack=True if is_training else False,
            use_custom_packing_ops=True if is_training else False),
        is_training=is_training,
        input_random_seed=(None if is_training else 4321),
        # eval_loop_num_batches=(1 if is_training else 5),
        batch_size=batch_size,
        reset_for_eval=False if is_training else True)
    return p

  def datasets(self) -> List[base_input.BaseInput.HParams]:
    """Returns a list of dataset parameters."""
    return [
        self._dataset_common(is_training=True),
        self._dataset_common(is_training=False)
    ]


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

  # optimizer related
  LEARNING_RATE = 1e-3
  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.99
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

  def task(self) -> base_task.BaseTask.HParams:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes
    model_p.lm.packed_input = self.PACKED_INPUT

    if self.USE_REPEATED_LAYER:
      stacked_transformer_tpl = model_p.lm.stacked_transformer_tpl.block
    else:
      stacked_transformer_tpl = model_p.lm.stacked_transformer_tpl
    transformer_layer_p = stacked_transformer_tpl.transformer_layer_params_tpl
    transformer_layer_p.tr_atten_tpl.use_bias = self.USE_BIAS

    lp = task_p.train.learner  # pytype: disable=attribute-error  # enable-nested-classes
    lp.loss_name = 'total_loss'
    lp.optimizer = optimizers.Adam.HParams(
        beta1=self.ADAM_BETA1,
        beta2=self.ADAM_BETA2,
        weight_decay=self.WEIGHT_DECAY,
        epsilon=self.ADAM_EPSILON,
        epsilon_root=self.ADAM_EPSILON_ROOT,
        clip_gradient_norm_to_value=self.CLIP_GRADIENT_NORM_TO_VALUE)
    lp.optimizer.learning_rate = self.LEARNING_RATE

    if self.LR_SCHEDULE == 'linear_rampup_exponential_decay':
      lp.optimizer.lr_schedule = (
          schedules.LinearRampupExponentialDecay.HParams(
              warmup_steps=self.LR_LRED_WARMUP,
              decay_start=self.LR_LRED_DECAY_START,
              decay_end=self.LR_LRED_DECAY_END,
              min_ratio=self.LR_LRED_MIN_RATIO,
              max=self.LR_LRED_MAX))
    elif self.LR_SCHEDULE == 'linear_rampup_cosine_decay':
      lp.optimizer.lr_schedule = (
          schedules.LinearRampupCosineDecay.HParams(
              warmup_steps=self.LR_COS_WARMUP,
              decay_start=self.LR_COS_DECAY_START,
              decay_end=self.LR_COS_DECAY_END,
              min_ratio=self.LR_COS_MIN_RATIO,
              max=self.LR_COS_MAX))
    else:
      raise NotImplementedError(f'Learning rate schedule {self.LR_SCHEDULE} is '
                                'not supported.')

    # pytype: enable=attribute-error  # enable-nested-classes
    return task_p


@experiment_registry.register
class LmCloudSpmdAdam(TransformerLmSpmdAdam, lm_cloud.SyntheticDataset):
  """Base config for an SPMD model."""

  NUM_LAYERS = 2
  MODEL_DIMS = 2048
  HIDDEN_DIMS = MODEL_DIMS * 4
  ACTIVATION = 'GELU'

  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # Sub-class has to specify a mesh.
  ICI_MESH_SHAPE = [1, 4, 2]


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
  ACTIVATION = 'GELU'

  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_FOR_MLPERF_200B
  CHECKPOINT_EVERY_N_STEPS = 1000

  # Sub-class has to specify a mesh.
  ICI_MESH_SHAPE = [1, 4, 2]


@experiment_registry.register
class C4SpmdGpt3Adam(C4SpmdAdam):
  r"""GPT-3 config for a decoder only transformer."""

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

  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  CHECKPOINT_EVERY_N_STEPS = 200
  CHECKPOINT_MAX_TO_KEEP = 2

  # 768 replicas with 1.5k global batch size
  PERCORE_BATCH_SIZE = 2
  ICI_MESH_SHAPE = [1, 64, 12]


@experiment_registry.register
class C4SpmdGpt3L16Adam(C4SpmdGpt3Adam):
  r"""a few layers of GPT-3 config for a decoder only transformer."""
  NUM_LAYERS = 16
  USE_REPEATED_LAYER = True
  # pad vocab to TPU-friendly size
  VOCAB_SIZE = 50304

  # 128 replicas with 128 global batch size using fp32
  PERCORE_BATCH_SIZE = 1
  ICI_MESH_SHAPE = [1, 32, 4]


@experiment_registry.register
class C4SpmdGpt3AdamHP(C4SpmdGpt3Adam):
  r"""GPT-3 config for a decoder only transformer."""
  NUM_LAYERS = 96
  USE_REPEATED_LAYER = True
  # pad vocab to TPU-friendly size
  VOCAB_SIZE = 50304

  # 1536 replicas with 1536 global batch size using fp32
  PERCORE_BATCH_SIZE = 1
  ICI_MESH_SHAPE = [1, 192, 8]
  # FPROP_DTYPE = jnp.float32
  FPROP_DTYPE = jnp.bfloat16

  # 512 replicas with 1536 global batch size using bf16
  # PERCORE_BATCH_SIZE = 3
  # ICI_MESH_SHAPE = [1, 128, 4]
  # FPROP_DTYPE = jnp.bfloat16

  # HPs
  ACTIVATION = 'GELU'
  LR_SCHEDULE = 'linear_rampup_exponential_decay'
  ADAM_CLIP_GRADIENT_NORM_TO_VALUE = 1.0

  CHECKPOINT_MAX_TO_KEEP = 10


class C4SpmdGpt3AdamOrgHP(C4SpmdGpt3Adam):
  r"""GPT-3 config with original HPs from the paper."""
  NUM_LAYERS = 96
  USE_REPEATED_LAYER = True

  # HPs
  ACTIVATION = 'GELU'
  LEARNING_RATE = 6e-5
  WEIGHT_DECAY = 0.1

  ADAM_BETA1 = 0.9
  ADAM_BETA2 = 0.95
  ADAM_EPSILON = 1e-8
  ADAM_CLIP_GRADIENT_NORM_TO_VALUE = 1.0

  LR_SCHEDULE = 'linear_rampup_cosine_decay'
  LR_COS_WARMUP = 265
  LR_COS_DECAY_START = 266
  LR_COS_DECAY_END = 108599
  LR_COS_MAX = 1.0
  LR_COS_MIN_RATIO = 0.1

  # Checkpoint
  EVAL_INTERVAL_STEPS = 100
  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_EVERY_N_STEPS = 100
  CHECKPOINT_MAX_TO_KEEP = 10

  def task(self) -> base_task.BaseTask.HParams:
    """Returns the task parameters."""
    task_p = super().task()
    model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes

    model_p.params_init = WeightInit.Gaussian(0.02)
    if self.USE_REPEATED_LAYER:
      stacked_transformer_tpl = model_p.lm.stacked_transformer_tpl.block
    else:
      stacked_transformer_tpl = model_p.lm.stacked_transformer_tpl
    transformer_layer_p = stacked_transformer_tpl.transformer_layer_params_tpl
    residual_dropout_p = transformer_layer_p.tr_fflayer_tpl.residual_dropout_tpl
    residual_dropout_p.params_init = WeightInit.Gaussian(
        1 / math.sqrt(self.NUM_LAYERS))

    return task_p


@experiment_registry.register
class C4SpmdGpt3AdamOrgHP1536Replicas(C4SpmdGpt3AdamOrgHP):
  r"""GPT-3 config in fp32 for 1536 replicas with 1536 global batch size."""
  PERCORE_BATCH_SIZE = 1
  ICI_MESH_SHAPE = [1, 192, 8]
  FPROP_DTYPE = jnp.float32


@experiment_registry.register
class C4SpmdGpt3AdamOrgHP512Replicas(C4SpmdGpt3AdamOrgHP):
  r"""GPT-3 config in bf16 for 512 replicas with 1536 global batch size."""
  PERCORE_BATCH_SIZE = 3
  ICI_MESH_SHAPE = [1, 64, 8]
  FPROP_DTYPE = jnp.bfloat16


@experiment_registry.register
class C4SpmdGpt3L16AdamOrgHP(C4SpmdGpt3AdamOrgHP):
  r"""Small GPT-3 config for a decoder only transformer."""

  NUM_LAYERS = 16
  # 64 replicas with 256 global batch size using fp32
  FPROP_DTYPE = jnp.float32
  PERCORE_BATCH_SIZE = 3
  ICI_MESH_SHAPE = [1, 16, 4]
