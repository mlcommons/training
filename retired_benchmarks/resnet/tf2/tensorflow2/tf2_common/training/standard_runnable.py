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
"""An abstraction that users can easily handle their custom training loops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import tensorflow as tf
from typing import Dict, Optional, Text

from tf2_common.training import runnable
from tf2_common.training import utils


@six.add_metaclass(abc.ABCMeta)
class StandardTrainable(runnable.AbstractTrainable):
  """Implements the standard functionality of AbstractTrainable APIs."""

  def __init__(self, use_tf_while_loop=True, use_tf_function=True):
    if use_tf_while_loop and not use_tf_function:
      raise ValueError("`use_tf_while_loop=True` and `use_tf_function=False` "
                       "is not supported")
    self.use_tf_while_loop = use_tf_while_loop
    self.use_tf_function = use_tf_function
    self.train_dataset = None
    self.train_iter = None
    self.train_loop_fn = None

  def _initialize_train_fn(self):
    if self.train_loop_fn is None:
      train_fn = self.train_step
      if self.use_tf_while_loop:
        self.train_loop_fn = utils.create_tf_while_loop_fn(train_fn)
      else:
        if self.use_tf_function:
          train_fn = tf.function(train_fn)
        self.train_loop_fn = utils.create_loop_fn(train_fn)

  def train(self,
            num_steps: Optional[tf.Tensor]) -> Optional[Dict[Text, tf.Tensor]]:
    """See base class."""
    if self.train_dataset is None:
      # Build train input dataset
      self.train_dataset = self.build_train_dataset()
      self.train_iter = tf.nest.map_structure(iter, self.train_dataset)

    self._initialize_train_fn()

    self.train_loop_begin()
    self.train_loop_fn(self.train_iter, num_steps)
    return self.train_loop_end()

  def train_loop_begin(self):
    """Called once at the beginning of the training loop.

    This is a good place to reset metrics that accumulate values over multiple
    steps of training.
    """
    pass

  @abc.abstractmethod
  def train_step(self, iterator):
    """Implements one step of training.

    What a "step" consists of is up to the implementer. If using distribution
    strategies, the call to this method should take place in the "cross-replica
    context" for generality, to allow e.g. multiple iterator dequeues and calls
    to `strategy.run`.

    Args:
      iterator: A tf.nest-compatible structure of tf.data Iterator or
        DistributedIterator.
    """
    pass

  def train_loop_end(self) -> Optional[Dict[Text, tf.Tensor]]:
    """Called at the end of the training loop.

    This is a good place to get metric results. The value returned from this
    function will be returned as-is from the train() method.

    Returns:
      The function may return a dictionary of `Tensors`, which will be
      written to logs and as TensorBoard summaries.
    """
    pass


@six.add_metaclass(abc.ABCMeta)
class StandardEvaluable(runnable.AbstractEvaluable):
  """Implements the standard functionality of AbstractEvaluable APIs."""

  def __init__(self, use_tf_function=True):
    self.eval_use_tf_function = use_tf_function
    self.eval_dataset = None
    self.eval_loop_fn = None

  @abc.abstractmethod
  def build_eval_dataset(self):
    """Builds the evaluation datasets.

    Returns:
      A tf.nest-compatible structure of tf.data.Dataset or DistributedDataset.
    """
    pass

  def _initialize_eval_fn(self):
    if self.eval_loop_fn is None:
      eval_fn = self.eval_step
      if self.eval_use_tf_function:
        eval_fn = tf.function(eval_fn)
      self.eval_loop_fn = utils.create_loop_fn(eval_fn)

  def evaluate(
      self, num_steps: Optional[tf.Tensor]) -> Optional[Dict[Text, tf.Tensor]]:
    """See base class."""
    if self.eval_dataset is None:
      # Build train input dataset
      self.eval_dataset = self.build_eval_dataset()

    self._initialize_eval_fn()
    # TODO(b/147718615): When async RPC is enabled in eager runtime, we make
    # eval iterator as a class member so it doesn't get destroyed when out of
    # the function scope.
    self.eval_iter = tf.nest.map_structure(iter, self.eval_dataset)

    self.eval_begin()
    self.eval_loop_fn(self.eval_iter, num_steps)
    return self.eval_end()

  def eval_begin(self):
    """Called once at the beginning of the evaluation.

    This is a good place to reset metrics that accumulate values over the entire
    evaluation.
    """
    pass

  @abc.abstractmethod
  def eval_step(self, iterator):
    """Implements one step of evaluation.

    What a "step" consists of is up to the implementer. If using distribution
    strategies, the call to this method should take place in the "cross-replica
    context" for generality, to allow e.g. multiple iterator dequeues and calls
    to `strategy.run`.

    Args:
      iterator: A tf.nest-compatible structure of tf.data Iterator or
        DistributedIterator.
    """
    pass

  def eval_end(self) -> Optional[Dict[Text, tf.Tensor]]:
    """Called at the end of the evaluation.

    This is a good place to get metric results. The value returned from this
    function will be returned as-is from the evaluate() method.

    Returns:
      The function may return a dictionary of `Tensors`, which will be
      written to logs and as TensorBoard summaries.
    """
    pass


@six.add_metaclass(abc.ABCMeta)
class StandardRunnableWithWarmup(StandardTrainable, StandardEvaluable):
  """A train and eval runnable that includes a warmup step."""

  def __init__(self, use_tf_while_loop=True, use_tf_function=True):
    StandardTrainable.__init__(self, use_tf_while_loop, use_tf_function)
    StandardEvaluable.__init__(self, use_tf_function)

    self.warmup_dataset = None
    self.warmup_iter = None

  @abc.abstractmethod
  def build_synthetic_dataset(self):
    """Builds the synthetic dataset for warmup.

    Returns:
      A tf.nest-compatible structure of tf.data.Dataset or DistributedDataset.
    """
    pass

  def warmup(self,
             num_steps: Optional[tf.Tensor]) -> Optional[Dict[Text, tf.Tensor]]:
    """Implements device warmup with multiple steps.

    This loop runs the input pipeline on synthetic data before training, thereby
    allowing XLA compilation and tf.function tracing before the dataset
    is accessed.

    Args:
      num_steps: A guideline for how many training steps to run. Note that it is
        up to the model what constitutes a "step" (this may involve more than
        one update to model parameters, e.g. if training a GAN).

    Returns:
      The function may return a dictionary of `Tensors`, which will be
      written to logs and as TensorBoard summaries.
    """
    if self.warmup_dataset is None:
      # Build warmup input dataset
      self.warmup_dataset = self.build_synthetic_dataset()
      self.warmup_iter = tf.nest.map_structure(iter, self.warmup_dataset)

    self._initialize_train_fn()
    self._initialize_eval_fn()
    self.warmup_loop_begin()
    self.train_loop_fn(self.warmup_iter, num_steps)
    self.eval_loop_fn(self.warmup_iter, num_steps)
    return self.warmup_loop_end()
