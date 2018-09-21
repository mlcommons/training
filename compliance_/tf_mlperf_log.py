# Copyright 2018 MLBenchmark Group. All Rights Reserved.
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
"""Convenience function for extracting the values for logging calls.

Because TensorFlow generally defers computation of values to a session run call,
it is impractical to log the values of tensors when they are defined. Instead,
the definition of a tensor is logged as normal using the log function in
mlperf_log.py and a tf.print statement helper function can be used to report
the relevant values as they are computed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import uuid

import tensorflow as tf


def log_deferred(op, log_id, repeat=False, every_n=1):
  """Helper method inserting compliance logging ops.

  Note: This helper is not guaranteed to be efficient, as it will insert ops
        and control dependencies. If this proves to be a bottleneck, submitters
        may wish to consider other methods such as extracting values from an
        .events file.

  Args:
    op: A tf op to be printed.
    log_id: a uuid provided by the logger in mlperf_log.py
    repeat: Should the value be logged multiple times?
    every_n: If repeat is True, with what frequency should the input op be '
             logged. If repeat is False, this argument is ignored.
  """

  prefix = ":::MLPv0.5.0 [{}]".format(log_id)
  if not repeat:
    return tf.Print(op, [tf.timestamp(), op], message=prefix, first_n=1)

  counter = tf.Variable(tf.zeros(shape=(), dtype=tf.int32) - 1)
  increment = tf.assign_add(counter, 1, use_locking=True)
  return tf.cond(
      tf.equal(tf.mod(increment, every_n), 0),
      lambda :tf.Print(op, [tf.timestamp(), op], message=prefix),
      lambda :op
  )


def _example():
  for kwargs in [dict(), dict(repeat=True), dict(repeat=True, every_n=2)]:
    op = tf.assign_add(tf.Variable(tf.zeros(shape=(), dtype=tf.int32) - 1), 1)
    op = log_deferred(op, str(uuid.uuid4()), **kwargs)
    init = [tf.local_variables_initializer(), tf.global_variables_initializer()]
    print("-" * 5)
    with tf.Session().as_default() as sess:
      sess.run(init)
      for _ in range(6):
        sess.run(op)


if __name__ == "__main__":
  _example()