# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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


import torch
from torch import nn

try:
  import apex
  #apex.amp.register_half_function(apex.normalization.fused_layer_norm, 'FusedLayerNorm')
  import apex.normalization
  #apex.amp.register_float_function(apex.normalization.FusedLayerNorm, 'forward')
  BertLayerNorm = apex.normalization.FusedLayerNorm
except ImportError:
  print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
  class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
      """Construct a layernorm module in the TF style (epsilon inside the square root).
      """
      super(BertLayerNorm, self).__init__()
      self.weight = nn.Parameter(torch.ones(hidden_size))
      self.bias = nn.Parameter(torch.zeros(hidden_size))
      self.variance_epsilon = eps

    def forward(self, x):
      u = x.mean(-1, keepdim=True)
      s = (x - u).pow(2).mean(-1, keepdim=True)
      x = (x - u) / torch.sqrt(s + self.variance_epsilon)
      return self.weight * x + self.bias

