# Copyright (c) 2019, Myrtle Software Limited. All rights reserved.
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
from torch.nn import Parameter
from mlperf import logging


def rnn(input_size, hidden_size, num_layers,
        forget_gate_bias=1.0, dropout=0.0,
        **kwargs):

    return LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        forget_gate_bias=forget_gate_bias,
        **kwargs,
    )

class LSTM(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout,
                 forget_gate_bias, weights_init_scale=1.0,
                 hidden_hidden_bias_scale=0.0, **kwargs):
        """Returns an LSTM with forget gate bias init to `forget_gate_bias`.

        Args:
            input_size: See `torch.nn.LSTM`.
            hidden_size: See `torch.nn.LSTM`.
            num_layers: See `torch.nn.LSTM`.
            dropout: See `torch.nn.LSTM`.
            forget_gate_bias: For each layer and each direction, the total value of
                to initialise the forget gate bias to.

        Returns:
            A `torch.nn.LSTM`.
        """
        super(LSTM, self).__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.dropout = torch.nn.Dropout(dropout) if dropout else None

        if forget_gate_bias is not None:
            for name, v in self.lstm.named_parameters():
                if "bias_ih" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size:2*hidden_size].fill_(forget_gate_bias)
                if "bias_hh" in name:
                    bias = getattr(self.lstm, name)
                    bias.data[hidden_size:2*hidden_size] *= float(hidden_hidden_bias_scale)

        for name, v in self.named_parameters():
            if 'weight' in name or 'bias' in name:
                v.data *= float(weights_init_scale)
        tensor_name = kwargs['tensor_name']
        logging.log_event(logging.constants.WEIGHTS_INITIALIZATION,
                          metadata=dict(tensor=tensor_name))


    def forward(self, x, h=None):
        x, h = self.lstm(x, h)
        if self.dropout:
            x = self.dropout(x)
        return x, h

