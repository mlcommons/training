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
from typing import Optional, Sequence

from jax import numpy as jnp
from paxml import base_experiment
from paxml import base_task
from paxml import tasks_lib
from praxis import asserts
from praxis import base_layer
from praxis import base_model
from praxis import layers
from praxis import optimizers
from praxis import py_utils
from praxis import schedules
from praxis.layers import models

NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit


def set_sharding_annotations_v1(
    task_p: base_task.BaseTask.HParams,
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
  model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes
  asserts.eq(len(ici_mesh_shape), 3)
  model_p.ici_mesh_shape = ici_mesh_shape
  if dcn_mesh_shape is not None:
    asserts.eq(len(dcn_mesh_shape), 3)
    model_p.dcn_mesh_shape = dcn_mesh_shape
  replica_axis = 'replica'
  data_axis = 'data'
  mdl_axis = 'mdl'
  mesh_axis_names = [replica_axis, data_axis, mdl_axis]
  task_p.train.inputs_split_mapping = NestedMap(  # pytype: disable=attribute-error  # enable-nested-classes
      map_1d=((replica_axis, data_axis),),
      map_2d=((replica_axis, data_axis), None))
  model_p.mesh_axis_names = mesh_axis_names
  if hasattr(model_p, 'lm'):
    model_p.lm = model_p.lm.cls.set_sharding_params_v1(
        model_p.lm,
        replica_axis=replica_axis,
        data_axis=data_axis,
        mdl_axis=mdl_axis,
        ici_mesh_shape=model_p.ici_mesh_shape,
        dcn_mesh_shape=model_p.dcn_mesh_shape,
        mesh_axis_names=mesh_axis_names,
        training_optimized=training_optimized)


def set_default_adafactor(task_p: base_task.BaseTask.HParams,
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
  lp = task_p.train.learner  # pytype: disable=attribute-error  # enable-nested-classes
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


def maybe_setup_moe_params(model_p: base_model.BaseModel.HParams) -> None:
  """Convert a FeedforwardLayer to a MoE Layer for StackedTransformer."""
  # pytype: disable=attribute-error  # enable-nested-classes
  if model_p.cls == layers.StackedTransformerRepeated:
    model_p = model_p.block

  if model_p.num_experts == 0:
    return model_p  # pytype: disable=bad-return-type  # enable-nested-classes

  ff_p = model_p.transformer_layer_params_tpl.tr_fflayer_tpl
  assert issubclass(ff_p.cls, layers.TransformerFeedForward)
  moe_p = model_p.moe_layer_tpl
  # pytype: enable=attribute-error  # enable-nested-classes
  # Copy over the base params.
  base_layer.BaseLayer.copy_base_hparams(ff_p, moe_p)
  # Copy over othe params.
  moe_p.name = ff_p.name
  moe_p.input_dims = ff_p.input_dims
  moe_p.hidden_dims = ff_p.hidden_dims
  moe_p.ln_tpl = ff_p.ln_tpl.Copy()
  moe_p.activation = ff_p.activation
  moe_p.relu_dropout_tpl = ff_p.relu_dropout_tpl.Copy()
  moe_p.relu_dropout_prob = ff_p.relu_dropout_prob
  moe_p.residual_dropout_tpl = ff_p.residual_dropout_tpl.Copy()
  moe_p.residual_dropout_prob = ff_p.residual_dropout_prob
  moe_p.add_skip_connection = ff_p.add_skip_connection
  moe_p.norm_policy = ff_p.norm_policy


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

  USE_REPEATED_LAYER = False
  TRAINABLE_POSITION_EMB = False
  TRAINABLE_PE_MAX_SEQ_LEN = 16 * 1024
  RELATIVE_BIAS = False
  USE_ROTARY_POSITION_EMB = False
  NORM_POLICY = 'pre'
  ENABLE_DCONV = False
  COMBINE_QKV = True
  ACTIVATION = 'RELU'
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

  # Sub-class has to specify a mesh.
  ICI_MESH_SHAPE = None
  # Default to a single slice
  DCN_MESH_SHAPE = [1, 1, 1]
  TRAINING_OPTIMIZED_SHARDING = True

  def task(self) -> base_task.BaseTask.HParams:
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
    task_p.model = models.LanguageModel.HParams(name='xformer_lm')
    model_p = task_p.model
    model_p.lm.packed_input = True
    model_p.lm.model_dims = self.MODEL_DIMS
    model_p.lm.vocab_size = self.VOCAB_SIZE

    softmax_init = WeightInit.Gaussian(1.0 / math.sqrt(self.MODEL_DIMS))
    # pytype: disable=attribute-error  # enable-nested-classes
    model_p.lm.softmax_tpl.params_init = softmax_init
    model_p.lm.softmax_tpl.scale_sqrt_depth = True
    model_p.lm.softmax_tpl.soft_cap_logits = self.SOFTMAX_CAP_LOGITS

    if self.TRAINABLE_POSITION_EMB:
      model_p.lm.position_emb_tpl = (
          layers.TrainablePositionalEmbedding.HParams(
              max_seq_length=self.TRAINABLE_PE_MAX_SEQ_LEN))

    stacked_transformer_tpl = layers.StackedTransformer.HParams()
    stacked_transformer_tpl.model_dims = self.MODEL_DIMS
    stacked_transformer_tpl.hidden_dims = self.HIDDEN_DIMS
    stacked_transformer_tpl.num_layers = self.NUM_LAYERS
    stacked_transformer_tpl.num_heads = num_heads
    stacked_transformer_tpl.dim_per_head = self.DIMS_PER_HEAD

    stacked_transformer_tpl.dropout_prob = self.DROPOUT_PROB
    transformer_layer_p = stacked_transformer_tpl.transformer_layer_params_tpl
    transformer_layer_p.tr_atten_tpl.atten_logit_cap = self.ATTEN_LOGIT_CAP
    transformer_layer_p.norm_policy = self.NORM_POLICY
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_atten_tpl.combine_qkv = self.COMBINE_QKV
    transformer_layer_p.tr_fflayer_tpl.activation = self.ACTIVATION
    transformer_layer_p.tr_atten_tpl.dconv_qkv = self.ENABLE_DCONV
    # pytype: enable=attribute-error  # enable-nested-classes

    # Only one of RELATIVE_BIAS or USE_ROTARY_POSITION_EMB can be True.
    assert (not self.RELATIVE_BIAS) or (not self.USE_ROTARY_POSITION_EMB)
    if self.RELATIVE_BIAS:
      transformer_layer_p.tr_atten_tpl.relative_bias_tpl = (
          layers.RelativeBias.HParams())
    if self.USE_ROTARY_POSITION_EMB:
      transformer_layer_p.tr_atten_tpl.use_rotary_position_emb = True

    if self.USE_REPEATED_LAYER:
      model_p.lm.stacked_transformer_tpl = (
          layers.StackedTransformerRepeated.HParams())
      stacked_transformer_tpl.num_layers = 1
      model_p.lm.stacked_transformer_tpl.block = stacked_transformer_tpl
      model_p.lm.stacked_transformer_tpl.x_times = self.NUM_LAYERS
      model_p.lm.stacked_transformer_tpl.checkpoint_policy = (
          self.CHECKPOINT_POLICY)
    else:
      model_p.lm.stacked_transformer_tpl = stacked_transformer_tpl

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

    if self.ICI_MESH_SHAPE is not None:
      set_sharding_annotations_v1(task_p, self.TRAINING_OPTIMIZED_SHARDING,
                                  self.ICI_MESH_SHAPE, self.DCN_MESH_SHAPE)
    maybe_setup_moe_params(model_p.lm.stacked_transformer_tpl)

    return task_p

