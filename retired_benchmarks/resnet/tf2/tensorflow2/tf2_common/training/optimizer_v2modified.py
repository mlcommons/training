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
"""Modified optimizer_v2 implementation enabling XLA across variable updates."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.optimizer_v2 import utils as optimizer_utils
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables as tf_variables

class OptimizerV2Modified(optimizer_v2.OptimizerV2):
  """This is a subclass optimizer that performs variable updates in
  Distribution Strategy replica context. OptimizerV2 base class is currently
  under refactoring and will have better support of this.

  Please refer to optimizer_v2.OptimizerV2 for more details regarding the APIs.
  """

  def __init__(self, name, use_experimental_compile=False, **kwargs):
    """Create a new Optimizer.

    Args:
      name: Optional name prefix for variables and ops created by the optimizer.
      use_experimental_compile: when set to True, use experimental_compile on
        the _distributed_apply function.
    """
    super(OptimizerV2Modified, self).__init__(name=name, **kwargs)
    self.use_experimental_compile = use_experimental_compile

  def apply_gradients(self,
                      grads_and_vars,
                      name=None,
                      experimental_aggregate_gradients=True):
    """Apply gradients to variables.

    Only the last two lines are different from optimizer_v2.OptimizerV2.

    Args:
      grads_and_vars: List of (gradient, variable) pairs.
      name: Optional name for the returned operation. Default to the name passed
        to the `Optimizer` constructor.
      experimental_aggregate_gradients: Whether to sum gradients from different
        replicas in the presense of `tf.distribute.Strategy`. If False, it's
        user responsibility to aggregate the gradients. Default to True.

    Returns:
      An `Operation` that applies the specified gradients. The `iterations`
      will be automatically increased by 1.

    Raises:
      TypeError: If `grads_and_vars` is malformed.
      ValueError: If none of the variables have gradients.
      RuntimeError: If called in cross-replica context.
    """
    # pylint: disable=protected-access
    grads_and_vars = optimizer_utils.filter_empty_gradients(grads_and_vars)
    # pylint: enable=protected-access
    var_list = [v for (_, v) in grads_and_vars]

    with ops.name_scope_v2(self._name):
      # Create iteration if necessary.
      with ops.init_scope():
        self._create_all_weights(var_list)

      if not grads_and_vars:
        # Distribution strategy does not support reducing an empty list of
        # gradients
        return control_flow_ops.no_op()

      if distribute_ctx.in_cross_replica_context():
        raise RuntimeError(
            "`apply_gradients() cannot be called in cross-replica context. "
            "Use `tf.distribute.Strategy.run` to enter replica "
            "context.")

      strategy = distribute_ctx.get_strategy()
      if (not experimental_aggregate_gradients and strategy and isinstance(
          strategy.extended,
          parameter_server_strategy.ParameterServerStrategyExtended)):
        raise NotImplementedError(
            "`experimental_aggregate_gradients=False is not supported for "
            "ParameterServerStrategy and CentralStorageStrategy")

      apply_state = self._prepare(var_list)
      if experimental_aggregate_gradients:
        grads_and_vars = self._transform_unaggregated_gradients(grads_and_vars)
        grads_and_vars = self._aggregate_gradients(grads_and_vars)
      grads_and_vars = self._transform_gradients(grads_and_vars)

      self._distributed_apply(None, grads_and_vars, name, apply_state)
      return self._iterations.assign_add(1, read_value=False)

  def _distributed_apply_org(self, distribution, grads_and_vars, name, apply_state):
    """`apply_gradients` using a `DistributionStrategy`.

    This is the _distributed_apply function in optimizer_v2,
    returning a list of ops.
    """

    def apply_grad_to_update_var(var, grad):
      """Apply gradient to variable."""
      if isinstance(var, ops.Tensor):
        raise NotImplementedError("Trying to update a Tensor ", var)

      apply_kwargs = {}
      if isinstance(grad, ops.IndexedSlices):
        if var.constraint is not None:
          raise RuntimeError(
              "Cannot use a constraint function on a sparse variable.")
        if "apply_state" in self._sparse_apply_args:
          apply_kwargs["apply_state"] = apply_state
        return self._resource_apply_sparse_duplicate_indices(
            grad.values, var, grad.indices, **apply_kwargs)

      if "apply_state" in self._dense_apply_args:
        apply_kwargs["apply_state"] = apply_state
      update_op = self._resource_apply_dense(grad, var, **apply_kwargs)
      if var.constraint is not None:
        with ops.control_dependencies([update_op]):
          return var.assign(var.constraint(var))
      else:
        return update_op

    update_ops = []
    with ops.name_scope(name or self._name, skip_on_eager=True):
      for grad, var in grads_and_vars:
        update_ops.append(apply_grad_to_update_var(var, grad))
      return control_flow_ops.group(*update_ops)

  def _distributed_apply(self, distribution, grads_and_vars, name, apply_state):
    if self.use_experimental_compile:
      self._distributed_apply_compile(distribution, grads_and_vars, name,
                                      apply_state)
    else:
      self._distributed_apply_org(distribution, grads_and_vars, name,
                                  apply_state)

  @tf.function(experimental_compile=True)
  def _distributed_apply_compile(self, distribution, grads_and_vars, name,
                                 apply_state):
    """This is a warpper, to return a tensor, making tf.func() happy."""
    self._distributed_apply_org(distribution, grads_and_vars,
                                name, apply_state)
    return tf.ones((), dtype=tf.bool)
