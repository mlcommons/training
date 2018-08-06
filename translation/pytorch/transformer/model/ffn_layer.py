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
"""Implementation of fully connected network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedFowardNetwork(nn.Module):
  """Fully connected feedforward network."""

  def __init__(self, hidden_size, filter_size, relu_dropout):
    super(FeedFowardNetwork, self).__init__()
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.relu_dropout = relu_dropout

    self.filter_dense_layer = nn.Linear(hidden_size, filter_size)
    self.output_dense_layer = nn.Linear(filter_size, hidden_size)

  def forward(self, x, padding=None, train=True):
    # Retrieve dynamically known shapes
    batch_size = x.shape[0]
    length = x.shape[1]

    if padding is not None:
      # Flatten padding to [batch_size*length]
      pad_mask = padding.reshape(-1)

      nonpad_ids = (pad_mask < 1e-9).nonzero().squeeze()

      # Reshape x to [batch_size*length, hidden_size] to remove padding
      x = x.reshape(-1, self.hidden_size)
      x = torch.index_select(x, dim=0, index=nonpad_ids)

      x = x.unsqueeze(dim=0)

    output = self.filter_dense_layer(x)
    output = nn.ReLU(inplace=True)(output)
    if train:
      output = F.dropout(output, p=self.relu_dropout, training=True)
    output = self.output_dense_layer(output)

    if padding is not None:
      output = output.squeeze(dim=0)
      output = x.new_zeros(batch_size*length, self.hidden_size).scatter_(
              0, nonpad_ids.repeat((self.hidden_size, 1)).t(), output)
      output = output.reshape(batch_size, length, self.hidden_size)
    return output

