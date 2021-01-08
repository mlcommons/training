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
from common.data.dataset import normalize_string


class DaliRnntIterator(object):
    """
    Returns batches of data for RNN-T training:
    preprocessed_signal, preprocessed_signal_length, transcript, transcript_length

    This iterator is not meant to be the entry point to Dali processing pipeline.
    Use DataLoader instead.
    """

    def __init__(self, dali_pipelines, transcripts, tokenizer, batch_size, shard_size, train_iterator: bool, normalize_transcripts=False):
        self.normalize_transcripts = normalize_transcripts
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        from nvidia.dali.plugin.pytorch import DALIGenericIterator
        from nvidia.dali.plugin.base_iterator import LastBatchPolicy

        # in train pipeline shard_size is set to divisable by batch_size, so PARTIAL policy is safe
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
        transcripts = torch.nn.utils.rnn.pad_sequence(transcripts, batch_first=True)

        return transcripts.cuda(), self.t_sizes[ids].cuda()

    def __next__(self):
        data = self.dali_it.__next__()
        audio, audio_shape = data[0]["audio"], data[0]["audio_shape"][:, 1]
        audio = audio[:, :, :audio_shape.max()] # the last batch
        transcripts, transcripts_lengths = self._gen_transcripts(data[0]["label"])
        return audio, audio_shape, transcripts, transcripts_lengths

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self


# TODO: refactor
class SyntheticDataIterator(object):
    def __init__(self, batch_size, nfeatures, feat_min=-5., feat_max=0., txt_min=0., txt_max=23., feat_lens_max=1760,
                 txt_lens_max=231, regenerate=False):
        """
        Args:
            batch_size
            nfeatures: number of features for melfbanks
            feat_min: minimum value in `feat` tensor, used for randomization
            feat_max: maximum value in `feat` tensor, used for randomization
            txt_min: minimum value in `txt` tensor, used for randomization
            txt_max: maximum value in `txt` tensor, used for randomization
            regenerate: If True, regenerate random tensors for every iterator step.
                        If False, generate them only at start.
        """
        self.batch_size = batch_size
        self.nfeatures = nfeatures
        self.feat_min = feat_min
        self.feat_max = feat_max
        self.feat_lens_max = feat_lens_max
        self.txt_min = txt_min
        self.txt_max = txt_max
        self.txt_lens_max = txt_lens_max
        self.regenerate = regenerate

        if not self.regenerate:
            self.feat, self.feat_lens, self.txt, self.txt_lens = self._generate_sample()

    def _generate_sample(self):
        feat = (self.feat_max - self.feat_min) * np.random.random_sample(
            (self.batch_size, self.nfeatures, self.feat_lens_max)) + self.feat_min
        feat_lens = np.random.randint(0, int(self.feat_lens_max) - 1, size=self.batch_size)
        txt = (self.txt_max - self.txt_min) * np.random.random_sample(
            (self.batch_size, self.txt_lens_max)) + self.txt_min
        txt_lens = np.random.randint(0, int(self.txt_lens_max) - 1, size=self.batch_size)
        return torch.Tensor(feat).cuda(), \
               torch.Tensor(feat_lens).cuda(), \
               torch.Tensor(txt).cuda(), \
               torch.Tensor(txt_lens).cuda()

    def __next__(self):
        if self.regenerate:
            return self._generate_sample()
        return self.feat, self.feat_lens, self.txt, self.txt_lens

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self
