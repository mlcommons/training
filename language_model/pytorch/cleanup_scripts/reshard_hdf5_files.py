# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

import glob
import h5py
import multiprocessing
import numpy as np

hdf5_compression_method = None

input_path = 'hdf5'
input_files = glob.glob(input_path + '/part*', recursive=False)

print('n_input_shards =', len(input_files))

max_pred_per_seq = 76
seq_length = 512
n_output_shards = 1472
n_processes = 20*2

print('n_output_shards =', n_output_shards)

# Allocate input storage
f_input_ids = []
f_input_masks = []
f_segment_ids = []
f_masked_lm_positions = []
f_masked_lm_ids = []
f_next_sentence_labels = []

# Load all samples into memory (can be done in batches if necessary in future)
def loader(idx):
  ifile = input_files[idx]
  print("Opening:", ifile, " --  Progress:", idx+1, '/', len(input_files))
  h5_ifile = h5py.File(ifile, 'r')

  f_input_ids = h5_ifile['input_ids'][:].astype(np.int64)
  f_input_masks = h5_ifile['input_mask'][:].astype(np.int64)
  f_segment_ids = h5_ifile['segment_ids'][:].astype(np.int64)
  f_masked_lm_positions = h5_ifile['masked_lm_positions'][:].astype(np.int64)
  f_masked_lm_ids = h5_ifile['masked_lm_ids'][:].astype(np.int64)
  f_next_sentence_labels = h5_ifile['next_sentence_labels'][:].astype(np.int64)

  h5_ifile.close()
  
  return f_input_ids, f_input_masks, f_segment_ids, f_masked_lm_positions, f_masked_lm_ids, f_next_sentence_labels

for idx, ifile in enumerate(input_files):
  print("Opening:", ifile, " --  Progress:", idx+1, '/', len(input_files))
  h5_ifile = h5py.File(ifile, 'r')

  f_input_ids.append(h5_ifile['input_ids'][:])
  f_input_masks.append(h5_ifile['input_mask'][:])
  f_segment_ids.append(h5_ifile['segment_ids'][:])
  f_masked_lm_positions.append(h5_ifile['masked_lm_positions'][:])
  f_masked_lm_ids.append(h5_ifile['masked_lm_ids'][:])
  f_next_sentence_labels.append(h5_ifile['next_sentence_labels'][:])

  h5_ifile.close()
 
# Calculate index offsets (to access into list of np.ndarray's)
n_samples = 0
f_offsets = [0]

print("len(f_input_ids) =", len(f_input_ids))

for idx in range(len(f_input_ids) - 1):
  f_offsets.append(f_input_ids[idx].shape[0] + f_offsets[-1])
  n_samples += f_input_ids[idx].shape[0]

n_samples += f_input_ids[-1].shape[0]

f_offsets = np.array(f_offsets)

print("n_samples =", n_samples)

# Create random permutation
rand_perm = np.random.permutation(n_samples)

def f_retrieve(global_idx, f_ndarray):
  idx = np.abs(f_offsets - global_idx).argmin()
  if f_offsets[idx] > global_idx:
    idx -= 1
  
  local_idx = global_idx - f_offsets[idx]
  perm_idx = rand_perm[global_idx]

  pidx = np.abs(f_offsets - perm_idx).argmin()
  if f_offsets[pidx] > perm_idx:
      pidx -= 1
  
  return f_ndarray[idx][pidx]


# Find a "nominal" number of samples per shard (calculated to always go over by one shard size)
# Find excess samples in last shard and distribute removal of excess over first "N" shards (could be done over last, but it doesn't matter and math is easier this way)
#  (since 0 <= excess < nominal_shard_size, the max imbalance will be 1 sample to minimize the straggler effect)
n_sample_per_ofile_nominal = (n_samples + n_output_shards - 1) // n_output_shards
n_excess = n_output_shards * n_sample_per_ofile_nominal - n_samples  # Always a positive number


# Start writing out sequentially from random permutation (mapping -> rand input to seq. output to favor disk I/O)
# The expected program execution time will be time(sequential_read) + time(sequential_writes) + negligibile_permutation_cpu_time
# Space requirement is approx. equal to size_of_input_bytes + size_of_permutation_array_bytes (~130GB + ~10MB)
# This is much better than the 30 hours or so required by the previous sharding script and the >400GB memory footprint (expected input size ~130GB)
ofile_prefix = '1472_balanced/part_'
ofile_suffix = '_of_' + str(n_output_shards) + '.hdf5'

global_index_offset = 0
for i in range(n_output_shards):
  ofile = ofile_prefix + str(i) + ofile_suffix
  
  n_samples_in_this_shard = n_sample_per_ofile_nominal
  if i < n_excess:
    n_samples_in_this_shard -= 1
    
  print("shard index:", i, ", n_samples_in_this_shard =", n_samples_in_this_shard)
  
  with h5py.File(ofile, 'w') as f:
    o_input_ids = np.ndarray((n_samples_in_this_shard, seq_length))
    o_input_masks = np.ndarray((n_samples_in_this_shard, seq_length))
    o_segment_ids = np.ndarray((n_samples_in_this_shard, seq_length))
    o_masked_lm_positions = np.ndarray((n_samples_in_this_shard, max_pred_per_seq))
    o_masked_lm_ids = np.ndarray((n_samples_in_this_shard, max_pred_per_seq))
    o_next_sentence_labels = np.ndarray((n_samples_in_this_shard))
      
    for local_index in range(n_samples_in_this_shard):
      o_input_ids[local_index] = f_retrieve(global_index_offset + local_index, f_input_ids)
      o_input_masks[local_index] = f_retrieve(global_index_offset + local_index, f_input_masks) 
      o_segment_ids[local_index] = f_retrieve(global_index_offset + local_index, f_segment_ids)
      o_masked_lm_positions[local_index] = f_retrieve(global_index_offset + local_index, f_masked_lm_positions)
      o_masked_lm_ids[local_index] = f_retrieve(global_index_offset + local_index, f_masked_lm_ids)
      o_next_sentence_labels[local_index] = f_retrieve(global_index_offset + local_index, f_next_sentence_labels)
      
    f.create_dataset("input_ids",            data=o_input_ids,            dtype='i4', compression=hdf5_compression_method)
    f.create_dataset("input_mask",           data=o_input_masks,          dtype='i1', compression=hdf5_compression_method)
    f.create_dataset("segment_ids",          data=o_segment_ids,          dtype='i1', compression=hdf5_compression_method)
    f.create_dataset("masked_lm_positions",  data=o_masked_lm_positions,  dtype='i4', compression=hdf5_compression_method)
    f.create_dataset("masked_lm_ids",        data=o_masked_lm_ids,        dtype='i4', compression=hdf5_compression_method)
    f.create_dataset("next_sentence_labels", data=o_next_sentence_labels, dtype='i1', compression=hdf5_compression_method)
      
    global_index_offset += n_samples_in_this_shard
