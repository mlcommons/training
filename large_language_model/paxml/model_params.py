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

"""Base language model configurations."""

import math
import typing
from typing import Optional, Sequence, cast, Type

import fiddle as fdl
from jax import numpy as jnp
from paxml import base_experiment
from paxml import tasks_lib
from praxis import asserts
from praxis import base_layer
from praxis import base_model
from praxis import layers
from praxis import optimizers
from praxis import pax_fiddle
from praxis import py_utils
from praxis import schedules
from praxis.layers import activations
from praxis.layers import embedding_softmax
from praxis.layers import models
from praxis.layers import transformer_models

NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit


def set_sharding_annotations_v1(
    task_p: tasks_lib.SingleTask.HParams,
    training_optimized: bool,
    ici_mesh_shape: Sequence[int],
    dcn_mesh_shape: Optional[Sequence[int]] = None,
) -> None:
  """Sets the sharding annotations in the task config for the given mesh.

  Args:
    task_p: The task parameters to update with sharding annotations.
    training_optimized: A bool indicating whether sharding is optimized for
      training by saving activation memory between forward and backward passes.
    ici_mesh_shape: a 3D sequence representing the mesh shape for a slice.
    dcn_mesh_shape: a 3D sequence representing the mesh across slices, or None.
  """
  model_p = task_p.model
  asserts.eq(len(ici_mesh_shape), 3)
  model_p.ici_mesh_shape = ici_mesh_shape
  if dcn_mesh_shape is not None:
    asserts.eq(len(dcn_mesh_shape), 3)
    model_p.dcn_mesh_shape = dcn_mesh_shape
  replica_axis = 'replica'
  data_axis = 'data'
  mdl_axis = 'mdl'
  mesh_axis_names = [replica_axis, data_axis, mdl_axis]
  task_p.train.inputs_split_mapping = NestedMap(
      map_1d=((replica_axis, data_axis),),
      map_2d=((replica_axis, data_axis), None))
  model_p.mesh_axis_names = mesh_axis_names
  if hasattr(model_p, 'lm_tpl'):
    lm_cls = cast(
        Type[layers.TransformerLm], pax_fiddle.get_callable(model_p.lm_tpl)
    )
    model_p.lm_tpl = lm_cls.set_sharding_params_v1(
        model_p.lm_tpl,
        replica_axis=replica_axis,
        data_axis=data_axis,
        mdl_axis=mdl_axis,
        ici_mesh_shape=model_p.ici_mesh_shape,
        dcn_mesh_shape=model_p.dcn_mesh_shape,
        mesh_axis_names=mesh_axis_names,
        training_optimized=training_optimized,
    )


def set_default_adam(task_p: tasks_lib.SingleTask.HParams,
                     learning_rate: float,
                     weight_decay: float,
                     *,
                     warmup_steps: int = 4000,
                     decay_start: int = 4001,
                     decay_end: int = 300000) -> None:
  """Sets the default Adam optimizer settings in the model config.

  Args:
    task_p: The task parameters to update with optimizer specs.
    learning_rate: The learning rate to set.
    weight_decay: The weight_decay to set.
    warmup_steps: The number of warmup steps for the model.
    decay_start: The step at which to start decaying the learning rate.
    decay_end: The step at which to end the learning rate decay.
  """
  lp = task_p.train.learner
  lp.loss_name = 'total_loss'
  lp.optimizer = optimizers.Adam.HParams(
      beta1=0.9,
      beta2=0.99,
      weight_decay=weight_decay,
      clip_gradient_norm_to_value=5.0)
  lp.optimizer.learning_rate = learning_rate
  lp.optimizer.lr_schedule = (
      schedules.LinearRampupExponentialDecay.HParams(
          warmup_steps=warmup_steps,
          decay_start=decay_start,
          decay_end=decay_end,
          min_ratio=0.1,
          max=1.0))


def set_default_adafactor(task_p: tasks_lib.SingleTask.HParams,
                          learning_rate: float,
                          weight_decay: float,
                          *,
                          warmup_steps: int = 4000,
                          decay_start: int = 4001,
                          decay_end: int = 100000,
                          clip_gradient_norm_to_value: float = 5.0) -> None:
  """Sets the default AdaFactor optimizer settings in the task config.

  Args:
    task_p: The task parameters to update with optimizer specs.
    learning_rate: The learning rate to set.
    weight_decay: The weight_decay to set.
    warmup_steps: The number of warmup steps for the model.
    decay_start: The step at which to start decaying the learning rate.
    decay_end: The step at which to end the learning rate decay.
    clip_gradient_norm_to_value: clip_gradient_norm_to_value.
  """
  lp = task_p.train.learner
  lp.loss_name = 'total_loss'
  lp.optimizer = optimizers.ShardedAdafactor.HParams(
      decay_method='adam',
      beta1=0.9,
      decay_adam=0.99,
      weight_decay=weight_decay,
      clip_gradient_norm_to_value=clip_gradient_norm_to_value)
  lp.optimizer.learning_rate = learning_rate
  lp.optimizer.lr_schedule = (
      schedules.LinearRampupExponentialDecay.HParams(
          warmup_steps=warmup_steps,
          decay_start=decay_start,
          decay_end=decay_end,
          min_ratio=0.1,
          max=1.0))


def maybe_setup_moe_params(
    model_p: pax_fiddle.Config[base_layer.BaseLayer],
) -> None:
  """Convert a FeedforwardLayer to a MoE Layer for StackedTransformer."""
  # pytype: disable=attribute-error  # enable-nested-classes
  if fdl.get_callable(model_p) == layers.StackedTransformerRepeated:
    model_p = model_p.block

  if model_p.num_experts == 0:
    return

  ff_p = model_p.transformer_layer_params_tpl.tr_fflayer_tpl
  assert issubclass(fdl.get_callable(ff_p), layers.TransformerFeedForward)
  moe_p = model_p.moe_layer_tpl
  # pytype: enable=attribute-error  # enable-nested-classes
  # Copy over the base params.
  base_layer.BaseLayerApi.copy_base_hparams(ff_p, moe_p)
  # Copy over othe params.
  moe_p.name = ff_p.name
  moe_p.input_dims = ff_p.input_dims
  if not moe_p.hidden_dims:
    # We can generally use different hidden_dims for FFN and MoE
    #
    # We should not override if moe_p.hidden_dims is explicitly set already.
    moe_p.hidden_dims = ff_p.hidden_dims
  moe_p.ln_tpl = ff_p.ln_tpl.clone()
  moe_p.activation_tpl = ff_p.activation_tpl.clone()
  # TransformerFeedForwardMoe does not have use_gated_activation
  # moe_p.use_gated_activation = ff_p.use_gated_activation
  #
  # We never did wi_0 and wi_1 in the MoE layer even when we used GATED_GELU for
  # FFN.
  moe_p.relu_dropout_tpl = ff_p.relu_dropout_tpl.clone()
  moe_p.relu_dropout_prob = ff_p.relu_dropout_prob
  moe_p.residual_dropout_tpl = ff_p.residual_dropout_tpl.clone()
  moe_p.residual_dropout_prob = ff_p.residual_dropout_prob
  moe_p.add_skip_connection = ff_p.add_skip_connection
  moe_p.norm_policy = ff_p.norm_policy


class ClassificationModelAdam(base_experiment.BaseExperiment):
  """A simple MLP language model configuration using Adam."""
  NUM_LAYER = 8
  INPUT_DIM = 4096
  HIDDEN_DIM = 7168
  OUTPUT_DIM = 4096
  LEARNING_RATE = 1e-3
  WEIGHT_DECAY = 1e-3
  CHECKPOINT_EVERY_N_STEPS = 5
  SUMMARY_INTERVAL_STEPS = 5
  NUM_TRAIN_STEPS = 10
  MLP_WEIGHT_SHARDING = None
  SOFTMAX_WEIGHT_SHARDING = None

  # sub-class specify a mesh to use SPMD
  MESH_SHAPE = None
  TRAINING_OPTIMIZED_SHARDING = True

  def task(self) -> tasks_lib.SingleTask.HParams:
    task_p = tasks_lib.SingleTask.HParams(name='classification_task')
    task_p.model = pax_fiddle.Config(
        models.ClassificationMLPModel, name='classification_model'
    )
    model_p = task_p.model
    model_p.mlp_tpl.ff_tpl.input_dims = self.INPUT_DIM  # pytype: disable=attribute-error  # enable-nested-classes
    model_p.mlp_tpl.ff_tpl.output_dims = self.OUTPUT_DIM  # pytype: disable=attribute-error  # enable-nested-classes
    model_p.mlp_tpl.hidden_dims = self.HIDDEN_DIM
    model_p.mlp_tpl.num_layers = self.NUM_LAYER
    model_p.softmax_tpl.input_dims = self.INPUT_DIM
    model_p.softmax_tpl.num_classes = self.INPUT_DIM
    task_p.train.save_interval_steps = self.CHECKPOINT_EVERY_N_STEPS
    task_p.train.summary_interval_steps = self.SUMMARY_INTERVAL_STEPS
    model_p.ici_mesh_shape = self.MESH_SHAPE
    model_p.mesh_axis_names = ['x', 'y', 'z']
    model_p.softmax_tpl.weight_split_dims_mapping.wt = self.SOFTMAX_WEIGHT_SHARDING
    model_p.mlp_tpl.ici_mesh_shape = model_p.mesh_shape
    model_p.mlp_tpl.weight_split_dims_mapping.wt = self.MLP_WEIGHT_SHARDING
    set_sharding_annotations_v1(task_p, self.TRAINING_OPTIMIZED_SHARDING,
                                self.MESH_SHAPE)
    set_default_adam(task_p, self.LEARNING_RATE, self.WEIGHT_DECAY)
    task_p.train.num_train_steps = self.NUM_TRAIN_STEPS
    return task_p


class TransformerBertPmapAdam(base_experiment.BaseExperiment):
  """Base Pmap Transformer Bert configuration using Adam."""

  NUM_LAYERS = 4
  VOCAB_SIZE = 32000
  NUM_HEADS = 8
  MODEL_DIMS = 128
  HIDDEN_DIMS = MODEL_DIMS * 4
  DROPOUT_PROB = 0.0
  LEARNING_RATE = 1e-3
  WEIGHT_DECAY = 1e-3
  USE_REPEATED_LAYER = False
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_WITH_NO_BATCH_DIM
  ACTIVATION_CLS = activations.ReLU
  USE_GATED_ACTIVATION = False
  # Save a checkpoint every n steps.
  CHECKPOINT_EVERY_N_STEPS = 5000
  DECAY_END = 300000

  FORCE_MASK_GENERATION = False

  ENABLE_BFLOAT16 = True

  def task(self) -> tasks_lib.SingleTask.HParams:
    """Returns the task parameters."""
    task_p = tasks_lib.SingleTask.HParams(name='bert_task')
    task_p.model = pax_fiddle.Config(
        models.BertModel, name='bert_lm',
        force_mask_generation=self.FORCE_MASK_GENERATION)
    model_p = task_p.model
    model_p.lm_tpl.model_type = transformer_models.LanguageModelType.BIDIRECTIONAL
    model_p.lm_tpl.packed_input = True
    model_p.lm_tpl.model_dims = self.MODEL_DIMS
    model_p.lm_tpl.vocab_size = self.VOCAB_SIZE
    # pytype: disable=attribute-error  # enable-nested-classes
    model_p.lm_tpl.softmax_tpl.scale_sqrt_depth = True
    model_p.lm_tpl.softmax_tpl.soft_cap_logits = 30.0

    stacked_transformer_tpl = pax_fiddle.Config(layers.StackedTransformer)

    stacked_transformer_tpl.model_dims = self.MODEL_DIMS
    stacked_transformer_tpl.hidden_dims = self.HIDDEN_DIMS
    stacked_transformer_tpl.num_layers = self.NUM_LAYERS
    stacked_transformer_tpl.num_heads = self.NUM_HEADS
    stacked_transformer_tpl.dropout_prob = self.DROPOUT_PROB
    transformer_layer_p = (stacked_transformer_tpl.transformer_layer_params_tpl)
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = 50.0
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_fflayer_tpl.activation_tpl = pax_fiddle.Config(
        self.ACTIVATION_CLS
    )
    transformer_layer_p.tr_fflayer_tpl.use_gated_activation = (
        self.USE_GATED_ACTIVATION)

    if self.USE_REPEATED_LAYER:
      model_p.lm_tpl.stacked_transformer_tpl = pax_fiddle.Config(
          layers.StackedTransformerRepeated
      )
      stacked_transformer_tpl.num_layers = 1
      model_p.lm_tpl.stacked_transformer_tpl.block = stacked_transformer_tpl
      model_p.lm_tpl.stacked_transformer_tpl.x_times = self.NUM_LAYERS
      model_p.lm_tpl.stacked_transformer_tpl.checkpoint_policy = (
          self.CHECKPOINT_POLICY)
    else:
      model_p.lm_tpl.stacked_transformer_tpl = stacked_transformer_tpl

    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(self.MODEL_DIMS))
    model_p.lm_tpl.softmax_tpl.params_init = softmax_init
    # pytype: enable=attribute-error  # enable-nested-classes

    task_p.train.save_interval_steps = self.CHECKPOINT_EVERY_N_STEPS

    if self.ENABLE_BFLOAT16:
      model_p.fprop_dtype = jnp.bfloat16

    maybe_setup_moe_params(model_p.lm_tpl.stacked_transformer_tpl)

    set_default_adam(
        task_p, self.LEARNING_RATE, self.WEIGHT_DECAY, decay_end=self.DECAY_END)

    return task_p


class TransformerBertSpmdAdafactor(base_experiment.BaseExperiment):
  """Base SPMD Transformer Bert configuration using AdaFactor."""

  NUM_LAYERS = 32
  VOCAB_SIZE = 32000
  NUM_HEADS = 16
  MODEL_DIMS = 1024
  HIDDEN_DIMS = MODEL_DIMS * 4
  DROPOUT_PROB = 0.0
  LEARNING_RATE = 1e-3
  WEIGHT_DECAY = 1e-3
  USE_REPEATED_LAYER = False
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_DOT_WITH_NO_BATCH_DIM
  ENABLE_BFLOAT16 = False
  MASK_TOKEN_ID = 0
  DECAY_END = 100000

  ACTIVATION_CLS = activations.ReLU
  USE_GATED_ACTIVATION = False

  # Sub-class has to specify a mesh.
  MESH_SHAPE = None
  TRAINING_OPTIMIZED_SHARDING = True

  # Save a checkpoint every n steps.
  CHECKPOINT_EVERY_N_STEPS = 500
  CHECKPOINT_SAVE_MAX_TO_KEEP = 10

  def task(self) -> tasks_lib.SingleTask.HParams:
    """Returns the task parameters."""
    task_p = tasks_lib.SingleTask.HParams(name='bert_task')
    task_p.model = pax_fiddle.Config(models.BertModel, name='bert_lm')
    model_p = task_p.model
    model_p.mask_token_id = self.MASK_TOKEN_ID
    model_p.lm_tpl.model_type = transformer_models.LanguageModelType.BIDIRECTIONAL
    model_p.lm_tpl.packed_input = True
    model_p.lm_tpl.model_dims = self.MODEL_DIMS
    model_p.lm_tpl.vocab_size = self.VOCAB_SIZE
    # pytype: disable=attribute-error  # enable-nested-classes
    model_p.lm_tpl.softmax_tpl.scale_sqrt_depth = True
    model_p.lm_tpl.softmax_tpl.soft_cap_logits = 30.0

    stacked_transformer_tpl = pax_fiddle.Config(layers.StackedTransformer)
    stacked_transformer_tpl.model_dims = self.MODEL_DIMS
    stacked_transformer_tpl.hidden_dims = self.HIDDEN_DIMS
    stacked_transformer_tpl.num_layers = self.NUM_LAYERS
    stacked_transformer_tpl.num_heads = self.NUM_HEADS
    stacked_transformer_tpl.dropout_prob = self.DROPOUT_PROB
    transformer_layer_p = (stacked_transformer_tpl.transformer_layer_params_tpl)
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = 50.0
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_atten_tpl.combine_qkv = True
    transformer_layer_p.tr_fflayer_tpl.activation_tpl = pax_fiddle.Config(
        self.ACTIVATION_CLS
    )
    transformer_layer_p.tr_fflayer_tpl.use_gated_activation = (
        self.USE_GATED_ACTIVATION)

    if self.USE_REPEATED_LAYER:
      model_p.lm_tpl.stacked_transformer_tpl = pax_fiddle.Config(
          layers.StackedTransformerRepeated
      )
      stacked_transformer_tpl.num_layers = 1
      model_p.lm_tpl.stacked_transformer_tpl.block = stacked_transformer_tpl
      model_p.lm_tpl.stacked_transformer_tpl.x_times = self.NUM_LAYERS
      model_p.lm_tpl.stacked_transformer_tpl.checkpoint_policy = (
          self.CHECKPOINT_POLICY)
    else:
      model_p.lm_tpl.stacked_transformer_tpl = stacked_transformer_tpl

    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(self.MODEL_DIMS))
    model_p.lm_tpl.softmax_tpl.params_init = softmax_init
    # pytype: enable=attribute-error  # enable-nested-classes

    if self.ENABLE_BFLOAT16:
      model_p.fprop_dtype = jnp.bfloat16

    task_p.train.save_max_to_keep = self.CHECKPOINT_SAVE_MAX_TO_KEEP

    set_default_adafactor(
        task_p, self.LEARNING_RATE, self.WEIGHT_DECAY, decay_end=self.DECAY_END)

    task_p.train.save_interval_steps = self.CHECKPOINT_EVERY_N_STEPS

    maybe_setup_moe_params(model_p.lm_tpl.stacked_transformer_tpl)
    set_sharding_annotations_v1(task_p, self.TRAINING_OPTIMIZED_SHARDING,
                                self.MESH_SHAPE)

    return task_p


class TransformerLmPmapAdam(base_experiment.BaseExperiment):
  """Base Pmap Transformer LM configuration using Adam."""

  NUM_LAYERS = 32
  VOCAB_SIZE = 32000
  NUM_HEADS = 16
  MODEL_DIMS = 1024
  HIDDEN_DIMS = MODEL_DIMS * 4
  INPUT_DROPOUT_PROB = 0.0
  DROPOUT_PROB = 0.0
  LEARNING_RATE = 1e-3
  WEIGHT_DECAY = 1e-3
  USE_REPEATED_LAYER = False
  ACTIVATION_CLS = activations.ReLU
  USE_GATED_ACTIVATION = False
  DECAY_END = 300000
  REL_POS_EMB_DIM = None

  PACKED_INPUT = True
  ATTEN_LOGIT_CAP = 50.0
  USE_BIAS = False

  def task(self) -> tasks_lib.SingleTask.HParams:
    """Returns the task parameters."""
    task_p = tasks_lib.SingleTask.HParams(name='xformer_task')
    task_p.model = pax_fiddle.Config(models.LanguageModel, name='xformer_lm')
    model_p = task_p.model
    model_p.lm_tpl.packed_input = self.PACKED_INPUT
    model_p.lm_tpl.model_dims = self.MODEL_DIMS
    model_p.lm_tpl.vocab_size = self.VOCAB_SIZE
    # pytype: disable=attribute-error  # enable-nested-classes
    model_p.lm_tpl.softmax_tpl.scale_sqrt_depth = True

    stacked_transformer_tpl = pax_fiddle.Config(layers.StackedTransformer)
    stacked_transformer_tpl.model_dims = self.MODEL_DIMS
    stacked_transformer_tpl.hidden_dims = self.HIDDEN_DIMS
    stacked_transformer_tpl.num_layers = self.NUM_LAYERS
    stacked_transformer_tpl.num_heads = self.NUM_HEADS

    stacked_transformer_tpl.input_dropout_prob = self.INPUT_DROPOUT_PROB
    stacked_transformer_tpl.dropout_prob = self.DROPOUT_PROB
    transformer_layer_p = typing.cast(
        pax_fiddle.Config[layers.Transformer],
        stacked_transformer_tpl.transformer_layer_params_tpl,
    )
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = self.ATTEN_LOGIT_CAP
    transformer_layer_p.tr_atten_tpl.use_bias = self.USE_BIAS
    transformer_layer_p.tr_fflayer_tpl.activation_tpl = pax_fiddle.Config(
        self.ACTIVATION_CLS
    )
    transformer_layer_p.tr_fflayer_tpl.use_gated_activation = (
        self.USE_GATED_ACTIVATION)

    if self.REL_POS_EMB_DIM is not None:
      atten_xl_p = pax_fiddle.Config(layers.DotProductAttentionXL)
      atten_xl_p.copy_fields_from(transformer_layer_p.tr_atten_tpl)
      atten_xl_p.set(rel_pos_emb_dim=self.REL_POS_EMB_DIM)
      transformer_layer_p.tr_atten_tpl = atten_xl_p

    if self.USE_REPEATED_LAYER:
      model_p.lm_tpl.stacked_transformer_tpl = pax_fiddle.Config(
          layers.StackedTransformerRepeated
      )
      stacked_transformer_tpl.num_layers = 1
      model_p.lm_tpl.stacked_transformer_tpl.block = (stacked_transformer_tpl)
      model_p.lm_tpl.stacked_transformer_tpl.x_times = self.NUM_LAYERS
    else:
      model_p.lm_tpl.stacked_transformer_tpl = stacked_transformer_tpl

    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(self.MODEL_DIMS))
    model_p.lm_tpl.softmax_tpl.params_init = softmax_init
    # pytype: enable=attribute-error  # enable-nested-classes

    maybe_setup_moe_params(model_p.lm_tpl.stacked_transformer_tpl)
    set_default_adam(
        task_p, self.LEARNING_RATE, self.WEIGHT_DECAY, decay_end=self.DECAY_END)

    return task_p


class TransformerLmSpmdAdafactor(base_experiment.BaseExperiment):
  """Base SPMD Transformer LM configuration using Adafactor."""
  # architecture related
  NUM_LAYERS = 10
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = None
  MODEL_DIMS = 2048
  HIDDEN_DIMS = MODEL_DIMS * 4
  FPROP_DTYPE = jnp.bfloat16
  PACKED_INPUT = True

  USE_REPEATED_LAYER = False
  SEPARATE_EMBEDDING = False
  TRAINABLE_POSITION_EMB = False
  TRAINABLE_PE_MAX_SEQ_LEN = 16 * 1024
  RELATIVE_BIAS = False
  USE_ROTARY_POSITION_EMB = False
  NORM_POLICY = 'pre'
  ENABLE_DCONV = False
  COMBINE_QKV = True
  ACTIVATION_CLS = activations.ReLU
  USE_GATED_ACTIVATION = False
  DECAY_END = 100000

  # optimizer related
  DROPOUT_PROB = 0.0
  LEARNING_RATE = 2.5e-4
  CLIP_GRADIENT_NORM_TO_VALUE = 5.0
  WEIGHT_DECAY = 1e-3
  SOFTMAX_CAP_LOGITS = 30.0
  ATTEN_LOGIT_CAP = 50.0
  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # checkpoint
  CHECKPOINT_EVERY_N_STEPS = 5000
  SUMMARY_INTERVAL_STEPS = 100
  CHECKPOINT_MAX_TO_KEEP = 10
  EVAL_INTERVAL_STEPS = 100

  # Sub-class has to specify a mesh.
  ICI_MESH_SHAPE = None
  # Default to a single slice
  DCN_MESH_SHAPE = [1, 1, 1]
  TRAINING_OPTIMIZED_SHARDING = True

  def task(self) -> tasks_lib.SingleTask.HParams:
    """Returns the task parameters."""
    if self.DIMS_PER_HEAD is not None:
      if self.NUM_HEADS is None:
        assert self.MODEL_DIMS % self.DIMS_PER_HEAD == 0
        num_heads = int(self.MODEL_DIMS / self.DIMS_PER_HEAD)
      else:
        assert self.MODEL_DIMS == self.NUM_HEADS * self.DIMS_PER_HEAD
        num_heads = self.NUM_HEADS
    else:
      assert self.NUM_HEADS is not None
      num_heads = self.NUM_HEADS

    task_p = tasks_lib.SingleTask.HParams(name='xformer_task')
    task_p.model = pax_fiddle.Config(models.LanguageModel, name='xformer_lm')
    model_p = task_p.model
    model_p.lm_tpl.packed_input = self.PACKED_INPUT
    model_p.lm_tpl.model_dims = self.MODEL_DIMS
    model_p.lm_tpl.vocab_size = self.VOCAB_SIZE

    if self.SEPARATE_EMBEDDING:
      model_p.lm_tpl.separate_embedding_tpl = pax_fiddle.Config(
          layers.Embedding
      )
      model_p.lm_tpl.softmax_tpl = pax_fiddle.Config(layers.FullSoftmax)

    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(self.MODEL_DIMS))
    # pytype: disable=attribute-error  # enable-nested-classes
    model_p.lm_tpl.softmax_tpl.params_init = softmax_init
    if self.SEPARATE_EMBEDDING:
      model_p.lm_tpl.separate_embedding_tpl.scale_sqrt_depth = True
    else:
      model_p.lm_tpl.softmax_tpl.scale_sqrt_depth = True
    model_p.lm_tpl.softmax_tpl.soft_cap_logits = self.SOFTMAX_CAP_LOGITS

    if self.TRAINABLE_POSITION_EMB:
      model_p.lm_tpl.position_emb_tpl = pax_fiddle.Config(
          layers.TrainablePositionalEmbedding,
          max_seq_length=self.TRAINABLE_PE_MAX_SEQ_LEN,
      )

    stacked_transformer_tpl = pax_fiddle.Config(layers.StackedTransformer)
    stacked_transformer_tpl.model_dims = self.MODEL_DIMS
    stacked_transformer_tpl.hidden_dims = self.HIDDEN_DIMS
    stacked_transformer_tpl.num_layers = self.NUM_LAYERS
    stacked_transformer_tpl.num_heads = num_heads
    stacked_transformer_tpl.dim_per_head = self.DIMS_PER_HEAD

    stacked_transformer_tpl.dropout_prob = self.DROPOUT_PROB
    transformer_layer_p = typing.cast(
        pax_fiddle.Config[layers.Transformer],
        stacked_transformer_tpl.transformer_layer_params_tpl,
    )
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = self.ATTEN_LOGIT_CAP
    transformer_layer_p.norm_policy = self.NORM_POLICY
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_atten_tpl.combine_qkv = self.COMBINE_QKV
    transformer_layer_p.tr_fflayer_tpl.activation_tpl = pax_fiddle.Config(
        self.ACTIVATION_CLS
    )
    transformer_layer_p.tr_fflayer_tpl.use_gated_activation = (
        self.USE_GATED_ACTIVATION)
    transformer_layer_p.tr_atten_tpl.dconv_qkv = self.ENABLE_DCONV
    # pytype: enable=attribute-error  # enable-nested-classes

    # Only one of RELATIVE_BIAS or USE_ROTARY_POSITION_EMB can be True.
    assert (not self.RELATIVE_BIAS) or (not self.USE_ROTARY_POSITION_EMB)
    if self.RELATIVE_BIAS:
      transformer_layer_p.tr_atten_tpl.relative_bias_tpl = pax_fiddle.Config(
          layers.RelativeBias
      )
    if self.USE_ROTARY_POSITION_EMB:
      transformer_layer_p.tr_atten_tpl.use_rotary_position_emb = True

    if self.USE_REPEATED_LAYER:
      model_p.lm_tpl.stacked_transformer_tpl = pax_fiddle.Config(
          layers.StackedTransformerRepeated
      )
      stacked_transformer_tpl.num_layers = 1
      model_p.lm_tpl.stacked_transformer_tpl.block = stacked_transformer_tpl
      model_p.lm_tpl.stacked_transformer_tpl.x_times = self.NUM_LAYERS
      model_p.lm_tpl.stacked_transformer_tpl.checkpoint_policy = (
          self.CHECKPOINT_POLICY)
    else:
      model_p.lm_tpl.stacked_transformer_tpl = stacked_transformer_tpl

    # Enable bf16.
    model_p.fprop_dtype = self.FPROP_DTYPE

    set_default_adafactor(
        task_p,
        self.LEARNING_RATE,
        self.WEIGHT_DECAY,
        decay_end=self.DECAY_END,
        clip_gradient_norm_to_value=self.CLIP_GRADIENT_NORM_TO_VALUE)

    task_p.train.save_interval_steps = self.CHECKPOINT_EVERY_N_STEPS
    task_p.train.summary_interval_steps = self.SUMMARY_INTERVAL_STEPS
    task_p.train.save_max_to_keep = self.CHECKPOINT_MAX_TO_KEEP
    task_p.train.eval_interval_steps = self.EVAL_INTERVAL_STEPS

    if self.ICI_MESH_SHAPE is not None:
      set_sharding_annotations_v1(task_p, self.TRAINING_OPTIMIZED_SHARDING,
                                  self.ICI_MESH_SHAPE, self.DCN_MESH_SHAPE)
    maybe_setup_moe_params(model_p.lm_tpl.stacked_transformer_tpl)

    return task_p


class TransformerLmSpmdPipelineAdafactor(TransformerLmSpmdAdafactor):
  """Base SPMD pipelined Transformer LM configuration using Adafactor."""
  # architecture related
  NUM_LAYERS = 10
  VOCAB_SIZE = 32000
  DIMS_PER_HEAD = 128
  NUM_HEADS = None
  MODEL_DIMS = 2048
  HIDDEN_DIMS = MODEL_DIMS * 4
  FPROP_DTYPE = jnp.bfloat16

  # Default these flags to False as we already have a loop over stages.
  USE_REPEATED_LAYER = False
  SEPARATE_EMBEDDING = False
  TRAINABLE_POSITION_EMB = False
  TRAINABLE_PE_MAX_SEQ_LEN = 16 * 1024
  RELATIVE_BIAS = False
  USE_ROTARY_POSITION_EMB = False
  NORM_POLICY = 'pre'
  ENABLE_DCONV = False
  COMBINE_QKV = True
  ACTIVATION_CLS = activations.ReLU
  USE_GATED_ACTIVATION = False

  # optimizer related
  DROPOUT_PROB = 0.0
  LEARNING_RATE = 2.5e-4
  CLIP_GRADIENT_NORM_TO_VALUE = 5.0
  WEIGHT_DECAY = 1e-3
  SOFTMAX_CAP_LOGITS = 30.0
  ATTEN_LOGIT_CAP = 50.0
  DECAY_END = 100000
  # Autodiff remat.
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING

  # checkpoint
  CHECKPOINT_EVERY_N_STEPS = 5000
  SUMMARY_INTERVAL_STEPS = 100
  CHECKPOINT_MAX_TO_KEEP = 10

  # Profiler related
  PROFILER_NUM_STEPS = 2
  PROFILER_MIN_DURATION_SEC = 1
  PROFILER_CAPTURE_STEP = None

  # Pipeline related.
  NUM_STAGES = None
  CIRCULAR_REPEAT = 1
  PIPELINE_BROADCAST_INPUTS = False
  # One of the two need to be set.
  NUM_MICROBATCHES = None
  MICROBATCH_SIZE = None

  # Sub-class has to specify a mesh with shape [NUM_STAGES, replica, data, mdl]
  ICI_MESH_SHAPE = None
  # Default to a single slice
  DCN_MESH_SHAPE = [1, 1, 1, 1]
  # The actual 'data' dims used on embedding weights. Sometimes (e.g. DCN) we
  # may want to avoid using 'data' on transformer layers to avoid weight
  # allgather on microbatches, but want to use 'data' on the embedding weight
  # which is outside the pipeline.
  EMB_W_DATA_DIMS = 'data'
  # Whether to do input/output streaming across stages. This is typicall useful
  # for DCN.
  STREAM_IO = False

  def task(self) -> tasks_lib.SingleTask.HParams:
    """Returns the task parameters."""
    if self.DIMS_PER_HEAD is not None:
      if self.NUM_HEADS is None:
        assert self.MODEL_DIMS % self.DIMS_PER_HEAD == 0
        num_heads = int(self.MODEL_DIMS / self.DIMS_PER_HEAD)
      else:
        assert self.MODEL_DIMS == self.NUM_HEADS * self.DIMS_PER_HEAD
        num_heads = self.NUM_HEADS
    else:
      assert self.NUM_HEADS is not None
      num_heads = self.NUM_HEADS

    assert self.NUM_STAGES is not None
    assert self.NUM_LAYERS % (self.NUM_STAGES * self.CIRCULAR_REPEAT) == 0
    assert self.NUM_MICROBATCHES is not None or self.MICROBATCH_SIZE is not None
    assert self.ICI_MESH_SHAPE is not None and len(self.ICI_MESH_SHAPE) == 4
    assert self.DCN_MESH_SHAPE is not None and len(self.DCN_MESH_SHAPE) == 4
    assert self.ICI_MESH_SHAPE[0] * self.DCN_MESH_SHAPE[0] == self.NUM_STAGES

    task_p = tasks_lib.SingleTask.HParams(name='xformer_task')
    task_p.model = pax_fiddle.Config(models.LanguageModel, name='xformer_lm')
    model_p = task_p.model
    model_p.lm_tpl.packed_input = True
    model_p.lm_tpl.model_dims = self.MODEL_DIMS
    model_p.lm_tpl.vocab_size = self.VOCAB_SIZE

    if self.SEPARATE_EMBEDDING:
      model_p.lm_tpl.separate_embedding_tpl = pax_fiddle.Config(
          layers.Embedding
      )
      model_p.lm_tpl.softmax_tpl = pax_fiddle.Config(layers.FullSoftmax)

    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(self.MODEL_DIMS))
    # pytype: disable=attribute-error  # enable-nested-classes
    model_p.lm_tpl.softmax_tpl.params_init = softmax_init
    if self.SEPARATE_EMBEDDING:
      model_p.lm_tpl.separate_embedding_tpl.scale_sqrt_depth = True
    else:
      model_p.lm_tpl.softmax_tpl.scale_sqrt_depth = True
    model_p.lm_tpl.softmax_tpl.soft_cap_logits = self.SOFTMAX_CAP_LOGITS

    if self.TRAINABLE_POSITION_EMB:
      model_p.lm_tpl.position_emb_tpl = pax_fiddle.Config(
          layers.TrainablePositionalEmbedding,
          max_seq_length=self.TRAINABLE_PE_MAX_SEQ_LEN,
      )

    stacked_transformer_tpl = pax_fiddle.Config(layers.StackedTransformer)
    stacked_transformer_tpl.model_dims = self.MODEL_DIMS
    stacked_transformer_tpl.hidden_dims = self.HIDDEN_DIMS
    stacked_transformer_tpl.num_layers = self.NUM_LAYERS // (
        self.NUM_STAGES * self.CIRCULAR_REPEAT)
    stacked_transformer_tpl.num_heads = num_heads
    stacked_transformer_tpl.dim_per_head = self.DIMS_PER_HEAD

    stacked_transformer_tpl.dropout_prob = self.DROPOUT_PROB
    transformer_layer_p = typing.cast(
        pax_fiddle.Config[layers.Transformer],
        stacked_transformer_tpl.transformer_layer_params_tpl,
    )
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = self.ATTEN_LOGIT_CAP
    transformer_layer_p.norm_policy = self.NORM_POLICY
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_atten_tpl.combine_qkv = self.COMBINE_QKV
    transformer_layer_p.tr_fflayer_tpl.activation_tpl = pax_fiddle.Config(
        self.ACTIVATION_CLS
    )
    transformer_layer_p.tr_fflayer_tpl.use_gated_activation = (
        self.USE_GATED_ACTIVATION)
    transformer_layer_p.tr_atten_tpl.dconv_qkv = self.ENABLE_DCONV
    # pytype: enable=attribute-error  # enable-nested-classes

    # Only one of RELATIVE_BIAS or USE_ROTARY_POSITION_EMB can be True.
    assert (not self.RELATIVE_BIAS) or (not self.USE_ROTARY_POSITION_EMB)
    if self.RELATIVE_BIAS:
      transformer_layer_p.tr_atten_tpl.relative_bias_tpl = pax_fiddle.Config(
          layers.RelativeBias
      )
    if self.USE_ROTARY_POSITION_EMB:
      transformer_layer_p.tr_atten_tpl.use_rotary_position_emb = True

    if self.USE_REPEATED_LAYER:
      stacked_transformer_tpl.num_layers = 1
      stacked_transformer_tpl = pax_fiddle.Config(
          layers.StackedTransformerRepeated, block=stacked_transformer_tpl
      )
      stacked_transformer_tpl.x_times = self.NUM_LAYERS // (
          self.NUM_STAGES * self.CIRCULAR_REPEAT)
      stacked_transformer_tpl.checkpoint_policy = self.CHECKPOINT_POLICY

    # Wrap it with a pipeline layer.
    model_p.lm_tpl.stacked_transformer_tpl = pax_fiddle.Config(
        layers.PipelinedTransformer,
        pipeline_stage=stacked_transformer_tpl,
        num_pipeline_stages=self.NUM_STAGES,
        circular_repeat=self.CIRCULAR_REPEAT,
        num_pipeline_microbatches=self.NUM_MICROBATCHES,
        pipeline_microbatch_size=self.MICROBATCH_SIZE,
        stream_io=self.STREAM_IO,
        checkpoint_policy=self.CHECKPOINT_POLICY,
        pipeline_broadcast_inputs=self.PIPELINE_BROADCAST_INPUTS,
    )

    # Enable bf16.
    model_p.fprop_dtype = self.FPROP_DTYPE

    set_default_adafactor(
        task_p,
        self.LEARNING_RATE,
        self.WEIGHT_DECAY,
        decay_end=self.DECAY_END,
        clip_gradient_norm_to_value=self.CLIP_GRADIENT_NORM_TO_VALUE)

    task_p.train.save_interval_steps = self.CHECKPOINT_EVERY_N_STEPS
    task_p.train.summary_interval_steps = self.SUMMARY_INTERVAL_STEPS
    task_p.train.save_max_to_keep = self.CHECKPOINT_MAX_TO_KEEP
    task_p.train.eval_interval_steps = self.EVAL_INTERVAL_STEPS
    task_p.train.profiler_num_steps = self.PROFILER_NUM_STEPS
    task_p.train.profiler_min_duration_sec = self.PROFILER_MIN_DURATION_SEC
    task_p.train.profiler_capture_step = self.PROFILER_CAPTURE_STEP
    maybe_setup_moe_params(
        model_p.lm_tpl.stacked_transformer_tpl.pipeline_stage)

    # Set up the sharding specifications.
    model_p.ici_mesh_shape = self.ICI_MESH_SHAPE
    model_p.dcn_mesh_shape = self.DCN_MESH_SHAPE
    stage_axis = 'stage'
    replica_axis = 'replica'
    data_axis = 'data'
    mdl_axis = 'mdl'
    mesh_axis_names = [stage_axis, replica_axis, data_axis, mdl_axis]
    model_p.mesh_axis_names = mesh_axis_names

    # Set in-stage layer shardings.
    lm_cls = cast(
        Type[layers.TransformerLm], pax_fiddle.get_callable(model_p.lm_tpl)
    )
    model_p.lm_tpl = lm_cls.set_sharding_params_v1(
        model_p.lm_tpl,
        replica_axis=replica_axis,
        data_axis=data_axis,
        mdl_axis=mdl_axis,
        ici_mesh_shape=model_p.ici_mesh_shape,
        dcn_mesh_shape=model_p.dcn_mesh_shape,
        mesh_axis_names=mesh_axis_names,
        training_optimized=self.TRAINING_OPTIMIZED_SHARDING,
    )

    # Include stage_axis in input partitioning to allow full data parallelism in
    # embedding layers.
    batch_dims = (stage_axis, replica_axis, data_axis)
    task_p.train.inputs_split_mapping = NestedMap(
        map_1d=(batch_dims,), map_2d=(batch_dims, None))

    # Run softmax/embedding in data parallelism across all cores.
    softmax_p = model_p.lm_tpl.softmax_tpl
    if self.SEPARATE_EMBEDDING:
      embedding_p = model_p.lm_tpl.separate_embedding_tpl
    else:
      embedding_p = model_p.lm_tpl.softmax_tpl
    embedding_p.activation_split_dims_mapping.emb_out_split_dims_mapping = [
        batch_dims,
        None,
        mdl_axis,
    ]
    embedding_p.activation_split_dims_mapping.out = [batch_dims, None, mdl_axis]
    if (
        fdl.get_callable(softmax_p)
        == embedding_softmax.GShardSharedEmbeddingSoftmax
    ):
      # Softmax weight is of shape [vocab_size, input_dim].
      softmax_p.weight_split_dims_mapping.wt = [mdl_axis, self.EMB_W_DATA_DIMS]
    elif fdl.get_callable(softmax_p) in {
        embedding_softmax.SharedEmbeddingSoftmax,
        embedding_softmax.FullSoftmax,
    }:
      # Softmax weight is of shape [input_dim, vocab_size].
      softmax_p.weight_split_dims_mapping.wt = [self.EMB_W_DATA_DIMS, mdl_axis]
    else:
      raise NotImplementedError(
          f'softmax class {fdl.get_callable(softmax_p)} not supported'
      )
    if self.SEPARATE_EMBEDDING:
      embedding_p.weight_split_dims_mapping.wt = [
          self.EMB_W_DATA_DIMS,
          mdl_axis,
      ]

    pipeline_layer_p = model_p.lm_tpl.stacked_transformer_tpl
    pipeline_layer_p.weight_split_dims_mapping.stages = [stage_axis]
    # Match the final output sharding to softmax input sharding.
    pipeline_layer_p.activation_split_dims_mapping.final_out = [
        batch_dims, None, mdl_axis
    ]

    return task_p
