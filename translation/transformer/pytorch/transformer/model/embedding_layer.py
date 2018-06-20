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
"""Implementation of embedding layer with shared weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import model_utils


class EmbeddingSharedWeights(nn.Module):
  """Calculates input embeddings and pre-softmax linear with shared weights."""

  def __init__(self, vocab_size, hidden_size):
    super(EmbeddingSharedWeights, self).__init__()
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.shared_weights = nn.Parameter(torch.normal(mean=0.,
            std=(torch.tensor(float(hidden_size) ** -0.5).repeat(
                self.vocab_size, self.hidden_size))))

  def forward(self, x):
    """Get token embeddings of x.

    Args:
      x: An int64 tensor with shape [batch_size, length]
    Returns:
      embeddings: float32 tensor with shape [batch_size, length, embedding_size]
    """
    embeddings = torch.stack([torch.index_select(self.shared_weights, 0, index.long()) for index in x])

    # Scale embedding by the sqrt of the hidden size
    embeddings *= self.hidden_size ** 0.5

    # Create binary array of size [batch_size, length]
    # where 1 = padding, 0 = not padding
    padding = model_utils.get_padding(x)

    # Set all padding embedding values to 0
    embeddings *= (1 - padding).unsqueeze(dim=-1)
    return embeddings

  def linear(self, x):
    """Computes logits by running x through a linear layer.

    Args:
      x: A float32 tensor with shape [batch_size, length, hidden_size]
    Returns:
      float32 tensor with shape [batch_size, length, vocab_size].
    """
    batch_size = x.shape[0]
    length = x.shape[1]

    x = x.reshape(-1, self.hidden_size)
    logits = torch.matmul(x, self.shared_weights.t())

    return logits.reshape(batch_size, length, self.vocab_size)
