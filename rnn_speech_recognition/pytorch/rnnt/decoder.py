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

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import label_collate


class RNNTGreedyDecoder:
    """A greedy transducer decoder.

    Args:
        blank_symbol: See `Decoder`.
        model: Model to use for prediction.
        max_symbols_per_step: The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        cutoff_prob: Skip to next step in search if current highest character
            probability is less than this.
    """
    def __init__(self, blank_idx, max_symbols_per_step=30, max_symbol_per_sample=None):
        self.blank_idx = blank_idx
        assert max_symbols_per_step is None or max_symbols_per_step > 0
        self.max_symbols = max_symbols_per_step
        assert max_symbol_per_sample is None or max_symbol_per_sample > 0
        self.max_symbol_per_sample = max_symbol_per_sample
        self._SOS = -1   # start of sequence

    def _pred_step(self, model, label, hidden, device):
        if label == self._SOS:
            return model.predict(None, hidden, add_sos=False)

        label = label_collate([[label]]).to(device)
        return model.predict(label, hidden, add_sos=False)

    def _joint_step(self, model, enc, pred, log_normalize=False):
        logits = model.joint(enc, pred)[:, 0, 0, :]

        if log_normalize:
            probs = F.log_softmax(logits, dim=len(logits.shape) - 1)
            return probs
        else:
            return logits

    def decode(self, model, x, out_lens):
        """Returns a list of sentences given an input batch.

        Args:
            x: A tensor of size (batch, channels, features, seq_len)
            out_lens: list of int representing the length of each sequence
                output sequence.

        Returns:
            list containing batch number of sentences (strings).
        """
        model = getattr(model, 'module', model)
        with torch.no_grad():
            # Apply optional preprocessing

            logits, out_lens = model.encode(x, out_lens)

            output = []
            for batch_idx in range(logits.size(0)):
                inseq = logits[batch_idx, :, :].unsqueeze(1)
                logitlen = out_lens[batch_idx]
                sentence = self._greedy_decode(model, inseq, logitlen)
                output.append(sentence)

        return output

    def _greedy_decode(self, model, x, out_len):
        training_state = model.training
        model.eval()

        device = x.device

        hidden = None
        label = []
        for time_idx in range(out_len):
            if  self.max_symbol_per_sample is not None \
                and len(label) > self.max_symbol_per_sample:
                break
            f = x[time_idx, :, :].unsqueeze(0)

            not_blank = True
            symbols_added = 0

            while not_blank and (
                    self.max_symbols is None or
                    symbols_added < self.max_symbols):
                g, hidden_prime = self._pred_step(
                    model,
                    self._SOS if label == [] else label[-1],
                    hidden,
                    device
                )
                logp = self._joint_step(model, f, g, log_normalize=False)[0, :]

                # get index k, of max prob
                v, k = logp.max(0)
                k = k.item()

                if k == self.blank_idx:
                    not_blank = False
                else:
                    label.append(k)
                    hidden = hidden_prime
                symbols_added += 1

        model.train(training_state)
        return label

