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

import os

import numpy as np


def hash_list_of_strings(li):
    return str(abs(hash(''.join(li))))


class SimpleSampler:
    def __init__(self):
        self.file_list_path = None
        self.dataset_size = None

    def write_file_list(self, files):
        with open(self.file_list_path, 'w') as f:
            f.writelines(f'{name} {label}\n' for name, label in files)

    def get_file_list_path(self):
        assert self.file_list_path, 'File list not initialized. Run make_file_list first'
        return self.file_list_path

    def get_dataset_size(self):
        assert self.dataset_size, 'Dataset size not known. Run make_file_list first'
        return self.dataset_size

    def is_sampler_random(self):
        return False

    def process_output_files(self, output_files):
        self.dataset_size = len(output_files)
        return [ (path, entry['label']) for path, entry in output_files.items() ]

    def make_file_list(self, output_files, json_names):
        self.file_list_path = os.path.join(
            "/tmp",
            "rnnt_dali.file_list." + hash_list_of_strings(json_names)
        )
        self.write_file_list(self.process_output_files(output_files))


class BucketingSampler(SimpleSampler):
    def __init__(self, num_buckets, batch_size, num_workers, num_epochs, rng):
        super(BucketingSampler, self).__init__()
        self.rng = rng
        self.num_buckets = num_buckets
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers

    def process_output_files(self, output_files):
        names = list(output_files)
        lengths = [output_files[name]['duration'] for name in names]
        labels = np.array([output_files[name]['label'] for name in names])
        len_ids = np.argsort(lengths)
        buckets = np.array_split(len_ids, self.num_buckets)

        gbs = self.batch_size * self.num_workers
        shuffled_buckets = np.array([
            perm
            for _ in range(self.num_epochs)          # for every epoch
            for bucket in buckets                    # from every bucket
            for perm in self.rng.permutation(bucket) # pick samples in random order
        ])

        # drop last batch
        epochs = np.reshape(shuffled_buckets, [self.num_epochs, -1])
        to_drop = epochs.shape[1] - (epochs.shape[1] // gbs * gbs)
        for epoch in epochs:
            dropped_idxs = self.rng.choice(epochs.shape[1], to_drop, replace=False)
            if dropped_idxs is not None:
                epoch[dropped_idxs] = -1
        epochs = epochs[epochs != -1].reshape(self.num_epochs, -1)
        self.dataset_size = epochs.shape[1]

        epochs_iters_batch = np.reshape(epochs, [self.num_epochs, -1, gbs])

        # shuffle iterations in epochs perserving batches
        for epoch in epochs_iters_batch:
            self.rng.shuffle(epoch, axis=0)

        epochs_iters_batch_worker = np.reshape(
            epochs_iters_batch,
            [self.num_epochs, -1, self.batch_size, self.num_workers]
        )
        workers_epochs_iters_batch = np.moveaxis(epochs_iters_batch_worker, -1, 0)

        return [
            (names[i], labels[i])
            for i in workers_epochs_iters_batch.flatten()
        ]

    def is_sampler_random(self):
        return True

