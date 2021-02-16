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

from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlperf import logging

from common.rnn import rnn


class StackTime(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = int(factor)

    def forward(self, x, x_lens):
        # T, B, U
        seq = [x]
        for i in range(1, self.factor):
            tmp = torch.zeros_like(x)
            tmp[:-i, :, :] = x[i:, :, :]
            seq.append(tmp)
        # x_lens = torch.ceil(x_lens.float() / self.factor).int()
        x_lens = (x_lens.int() + self.factor - 1) // self.factor
        return torch.cat(seq, dim=2)[::self.factor, :, :], x_lens


class RNNT(nn.Module):
    """A Recurrent Neural Network Transducer (RNN-T).

    Args:
        in_features: Number of input features per step per batch.
        vocab_size: Number of output symbols (inc blank).
        forget_gate_bias: Total initialized value of the bias used in the
            forget gate. Set to None to use PyTorch's default initialisation.
            (See: http://proceedings.mlr.press/v37/jozefowicz15.pdf)
        batch_norm: Use batch normalization in encoder and prediction network
            if true.
        encoder_n_hidden: Internal hidden unit size of the encoder.
        encoder_rnn_layers: Encoder number of layers.
        pred_n_hidden:  Internal hidden unit size of the prediction network.
        pred_rnn_layers: Prediction network number of layers.
        joint_n_hidden: Internal hidden unit size of the joint network.
    """
    def __init__(self, n_classes, in_feats, enc_n_hid,
                 enc_pre_rnn_layers, enc_post_rnn_layers, enc_stack_time_factor,
                 enc_dropout, pred_dropout, joint_dropout,
                 pred_n_hid, pred_rnn_layers, joint_n_hid,
                 forget_gate_bias,
                 hidden_hidden_bias_scale=0.0, weights_init_scale=1.0,
                 enc_lr_factor=1.0, pred_lr_factor=1.0, joint_lr_factor=1.0):
        super(RNNT, self).__init__()

        self.enc_lr_factor = enc_lr_factor
        self.pred_lr_factor = pred_lr_factor
        self.joint_lr_factor = joint_lr_factor

        self.pred_n_hid = pred_n_hid

        pre_rnn_input_size = in_feats

        post_rnn_input_size = enc_stack_time_factor * enc_n_hid

        enc_mod = {}
        enc_mod["pre_rnn"] = rnn(input_size=pre_rnn_input_size,
                                 hidden_size=enc_n_hid,
                                 num_layers=enc_pre_rnn_layers,
                                 forget_gate_bias=forget_gate_bias,
                                 hidden_hidden_bias_scale=hidden_hidden_bias_scale,
                                 weights_init_scale=weights_init_scale,
                                 dropout=enc_dropout,
                                 tensor_name='pre_rnn',
                                )

        enc_mod["stack_time"] = StackTime(enc_stack_time_factor)

        enc_mod["post_rnn"] = rnn(input_size=post_rnn_input_size,
                                  hidden_size=enc_n_hid,
                                  num_layers=enc_post_rnn_layers,
                                  forget_gate_bias=forget_gate_bias,
                                  hidden_hidden_bias_scale=hidden_hidden_bias_scale,
                                  weights_init_scale=weights_init_scale,
                                  dropout=enc_dropout,
                                  tensor_name='post_rnn',
                                )

        self.encoder = torch.nn.ModuleDict(enc_mod)

        pred_embed = torch.nn.Embedding(n_classes - 1, pred_n_hid)
        logging.log_event(logging.constants.WEIGHTS_INITIALIZATION,
                          metadata=dict(tensor='pred_embed'))

        self.prediction = torch.nn.ModuleDict({
            "embed": pred_embed,
            "dec_rnn": rnn(
                input_size=pred_n_hid,
                hidden_size=pred_n_hid,
                num_layers=pred_rnn_layers,
                forget_gate_bias=forget_gate_bias,
                hidden_hidden_bias_scale=hidden_hidden_bias_scale,
                weights_init_scale=weights_init_scale,
                dropout=pred_dropout,
                tensor_name='dec_rnn',
            ),
        })

        self.joint_pred = torch.nn.Linear(
            pred_n_hid,
            joint_n_hid)
        logging.log_event(logging.constants.WEIGHTS_INITIALIZATION,
                          metadata=dict(tensor='joint_pred'))
        self.joint_enc = torch.nn.Linear(
            enc_n_hid,
            joint_n_hid)
        logging.log_event(logging.constants.WEIGHTS_INITIALIZATION,
                          metadata=dict(tensor='joint_enc'))

        self.joint_net = nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=joint_dropout),
            torch.nn.Linear(joint_n_hid, n_classes))
        logging.log_event(logging.constants.WEIGHTS_INITIALIZATION,
                          metadata=dict(tensor='joint_net'))

    def forward(self, x, x_lens, y, y_lens, state=None):
        # x: (B, channels, features, seq_len)
        y = label_collate(y)

        f, x_lens = self.encode(x, x_lens)

        g, _ = self.predict(y, state)
        out = self.joint(f, g)


        return out, x_lens

    def encode(self, x, x_lens):
        """
        Args:
            x: tuple of ``(input, input_lens)``. ``input`` has shape (T, B, I),
                ``input_lens`` has shape ``(B,)``.

        Returns:
            f: tuple of ``(output, output_lens)``. ``output`` has shape
                (B, T, H), ``output_lens``
        """
        x, _ = self.encoder["pre_rnn"](x, None)
        x, x_lens = self.encoder["stack_time"](x, x_lens)
        x, _ = self.encoder["post_rnn"](x, None)

        return x.transpose(0, 1), x_lens

    def predict(self, y, state=None, add_sos=True):
        """
        B - batch size
        U - label length
        H - Hidden dimension size
        L - Number of decoder layers = 2

        Args:
            y: (B, U)

        Returns:
            Tuple (g, hid) where:
                g: (B, U + 1, H)
                hid: (h, c) where h is the final sequence hidden state and c is
                    the final cell state:
                        h (tensor), shape (L, B, H)
                        c (tensor), shape (L, B, H)
        """
        if y is not None:
            # (B, U) -> (B, U, H)
            y = self.prediction["embed"](y)
        else:
            B = 1 if state is None else state[0].size(1)
            y = torch.zeros((B, 1, self.pred_n_hid)).to(
                device=self.joint_enc.weight.device,
                dtype=self.joint_enc.weight.dtype
            )

        # preprend blank "start of sequence" symbol
        if add_sos:
            B, U, H = y.shape
            start = torch.zeros((B, 1, H)).to(device=y.device, dtype=y.dtype)
            y = torch.cat([start, y], dim=1).contiguous()   # (B, U + 1, H)
        else:
            start = None   # makes del call later easier

        y = y.transpose(0, 1)#.contiguous()   # (U + 1, B, H)
        g, hid = self.prediction["dec_rnn"](y, state)
        g = g.transpose(0, 1)#.contiguous()   # (B, U + 1, H)
        del y, start, state
        return g, hid

    def joint(self, f, g):
        """
        f should be shape (B, T, H)
        g should be shape (B, U + 1, H)

        returns:
            logits of shape (B, T, U, K + 1)
        """
        # Combine the input states and the output states
        f = self.joint_enc(f)
        g = self.joint_pred(g)

        f = f.unsqueeze(dim=2)   # (B, T, 1, H)
        g = g.unsqueeze(dim=1)   # (B, 1, U + 1, H)

        res = self.joint_net(f + g)

        del f, g
        return res

    def param_groups(self, lr):
        chain_params = lambda *layers: chain(*[l.parameters() for l in layers])
        return [{'params': chain_params(self.encoder),
                 'lr': lr * self.enc_lr_factor},
                {'params': chain_params(self.prediction),
                 'lr': lr * self.pred_lr_factor},
                {'params': chain_params(self.joint_enc, self.joint_pred, self.joint_net),
                 'lr': lr * self.joint_lr_factor},
               ]


def label_collate(labels):
    """Collates the label inputs for the rnn-t prediction network.

    If `labels` is already in torch.Tensor form this is a no-op.

    Args:
        labels: A torch.Tensor List of label indexes or a torch.Tensor.

    Returns:
        A padded torch.Tensor of shape (batch, max_seq_len).
    """

    if isinstance(labels, torch.Tensor):
        return labels.type(torch.int64)
    if not isinstance(labels, (list, tuple)):
        raise ValueError(
            f"`labels` should be a list or tensor not {type(labels)}"
        )

    batch_size = len(labels)
    max_len = max(len(l) for l in labels)

    cat_labels = np.full((batch_size, max_len), fill_value=0.0, dtype=np.int32)
    for e, l in enumerate(labels):
        cat_labels[e, :len(l)] = l
    labels = torch.LongTensor(cat_labels)

    return labels
