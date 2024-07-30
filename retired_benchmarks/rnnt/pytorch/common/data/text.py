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

import sentencepiece as spm


class Tokenizer:
    def __init__(self, labels, sentpiece_model=None):
        """Converts transcript to a sequence of tokens.

        Args:
            labels (str): all possible output symbols
        """
        # For labels use vocab or load worpieces
        self.charset = labels
        self.use_sentpiece = (sentpiece_model is not None)
        if self.use_sentpiece:
            self.sentpiece = spm.SentencePieceProcessor(model_file=sentpiece_model)
            self.num_labels = len(self.sentpiece)
        else:
            self.num_labels = len(self.charset)
            self.label2ind = {lab: i for i, lab in enumerate(self.charset)}

    def tokenize(self, transcript):
        if self.use_sentpiece:
            inds = self.sentpiece.encode(transcript, out_type=int)
            assert 0 not in inds, '<unk> found during tokenization (OOV?)'
        else:
            inds = [self.label2ind[x]
                    for x in transcript if x in self.label2ind]
        return inds

    def detokenize(self, inds):
        if self.use_sentpiece:
            return self.sentpiece.decode(inds)
        else:
            return ''.join(self.charset[i] for i in inds)


