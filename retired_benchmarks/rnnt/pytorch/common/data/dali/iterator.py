# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import torch.distributed as dist
import numpy as np
from common.helpers import print_once
from common.text import _clean_text, punctuation_map


def normalize_string(s, charset, punct_map):
    """Normalizes string.

    Example:
        'call me at 8:00 pm!' -> 'call me at eight zero pm'
    """
    charset = set(charset)
    try:
        text = _clean_text(s, ["english_cleaners"], punct_map).strip()
        return ''.join([tok for tok in text if all(t in charset for t in tok)])
    except:
        print(f"WARNING: Normalizing failed: {s}")
        return None


class DaliRnntIterator(object):
    """
    Returns batches of data for RNN-T training:
    preprocessed_signal, preprocessed_signal_length, transcript, transcript_length

    This iterator is not meant to be the entry point to Dali processing pipeline.
    Use DataLoader instead.
    """

    def __init__(self, dali_pipelines, transcripts, tokenizer, batch_size, shard_size, pipeline_type, normalize_transcripts=False):
        self.normalize_transcripts = normalize_transcripts
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        from nvidia.dali.plugin.pytorch import DALIGenericIterator
        from nvidia.dali.plugin.base_iterator import LastBatchPolicy

        # in train pipeline shard_size is set to divisable by batch_size, so PARTIAL policy is safe
        if pipeline_type == 'val':
            self.dali_it = DALIGenericIterator(
                dali_pipelines, ["audio", "label", "audio_shape"], reader_name="Reader",
                dynamic_shape=True, auto_reset=True,
                last_batch_policy=LastBatchPolicy.PARTIAL)
        else:
            self.dali_it = DALIGenericIterator(
                dali_pipelines, ["audio", "label", "audio_shape"], size=shard_size,
                dynamic_shape=True, auto_reset=True, last_batch_padded=True,
                last_batch_policy=LastBatchPolicy.PARTIAL)

        self.tokenize(transcripts)

    def tokenize(self, transcripts):
        transcripts = [transcripts[i] for i in range(len(transcripts))]
        if self.normalize_transcripts:
            transcripts = [
                normalize_string(
                    t,
                    self.tokenizer.charset,
                    punctuation_map(self.tokenizer.charset)
                ) for t in transcripts
            ]
        transcripts = [self.tokenizer.tokenize(t) for t in transcripts]
        transcripts = [torch.tensor(t) for t in transcripts]
        self.tr = np.array(transcripts, dtype=object)
        self.t_sizes = torch.tensor([len(t) for t in transcripts], dtype=torch.int32)

    def _gen_transcripts(self, labels, normalize_transcripts: bool = True):
        """
        Generate transcripts in format expected by NN
        """
        ids = labels.flatten().numpy()
        transcripts = self.tr[ids]
        # Tensors are padded with 0. In `sentencepiece` we set it to <unk>,
        # because it cannot be disabled, and is absent in the data.
        # Note this is different from the RNN-T blank token (index 1023).
        transcripts = torch.nn.utils.rnn.pad_sequence(transcripts, batch_first=True)

        return transcripts.cuda(), self.t_sizes[ids].cuda()

    def __next__(self):
        data = self.dali_it.__next__()
        audio, audio_shape = data[0]["audio"], data[0]["audio_shape"][:, 1]
        if audio.shape[0] == 0:
            # empty tensor means, other GPUs got last samples from dataset
            # and this GPU has nothing to do; calling `__next__` raises StopIteration
            return self.dali_it.__next__()
        audio = audio[:, :, :audio_shape.max()] # the last batch
        transcripts, transcripts_lengths = self._gen_transcripts(data[0]["label"])
        return audio, audio_shape, transcripts, transcripts_lengths

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self


