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
"""Toolbox to reduce a large matrix into a smaller yet similar matrix.

Given a large matrix R of size (m, n), the toolbox produces a matrix R_hat which
is much smaller but is similar in the largest singular values.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.linalg

import skimage.transform as transform


def _closest_column_orthogonal_matrix(matrix):
  return np.matmul(matrix, np.linalg.inv(
      scipy.linalg.sqrtm(np.matmul(matrix.T, matrix))))


def resize_matrix(usv, num_rows, num_cols):
  """Apply algorith 2 in https://arxiv.org/pdf/1901.08910.pdf.

  Args:
    usv: matrix to reduce given in SVD form with the spectrum s in
      increasing order.
    num_rows: number of rows in the output matrix.
    num_cols: number of columns in the output matrix.
  Returns:
    A resized version of (u, s, v) whose non zero singular values will be
      identical to the largest singular values in s.
  """
  u, s, v = usv
  k = min(num_rows, num_cols)

  u_random_proj = transform.resize(u[:, :k], (num_rows, k))
  v_random_proj = transform.resize(v[:k, :], (k, num_cols))

  u_random_proj_orth = _closest_column_orthogonal_matrix(u_random_proj)
  v_random_proj_orth = _closest_column_orthogonal_matrix(v_random_proj.T).T

  return np.matmul(u_random_proj_orth,
                   np.matmul(np.diag(s[::-1][:k]), v_random_proj_orth))


def normalize_matrix(matrix):
  """Fold all values of the matrix into [0, 1]."""
  abs_matrix = np.abs(matrix.copy())
  return abs_matrix / abs_matrix.max()
