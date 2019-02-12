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
"""Toolbox for graph adjancency/incidence matrix analysis.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from scipy.sparse import linalg

flags.DEFINE_integer("max_iter_sparse_svd",
                     16,
                     "Maximum number of iterations in sparse "
                     "Singular Value Decomposition through power iterations.")

FLAGS = flags.FLAGS


def sparse_svd(sparse_matrix, num_values, max_iter):
  """Wrapper around SciPy's Singular Value Decomposition for sparse matrices.

  Args:
    sparse_matrix: a SciPy sparse matrix (typically large).
    num_values: the number of largest singular values to compute.
    max_iter: maximum number of iterations (>= 0) in the decomposition. If
      max_iter is None, runs FLAGS.max_iter_sparse_svd steps. If max_iter == 0,
      runs until convergence. Otherwise will run max_iter steps.

  Returns:
    A (u, s, v) tuple where s is an array entailing the singular values,
      and (u, v) the singular vector matrices. u is column orthogonal and
      v is row orthogonal. s is sorted in increasing order.
  """

  if num_values <= 0:
    raise ValueError("num_values should be > 0 but instead is %d." % num_values)

  if max_iter is not None and max_iter < 0:
    raise ValueError("max_iter should be >= 0 but instead is %d." % max_iter)

  if max_iter is None:
    max_iter = FLAGS.max_iter_sparse_svd
  elif not max_iter:
    max_iter = None

  u, s, v = linalg.svds(
      sparse_matrix, k=num_values, maxiter=max_iter,
      return_singular_vectors=True)

  return (u, s, v)

