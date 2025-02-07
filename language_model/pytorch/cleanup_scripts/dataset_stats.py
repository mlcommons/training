# NVIDIA

import h5py
import numpy as np

input_path = '/workspace/data_phase2/'
n_shards = 2048

input_files = [input_path + 'part_' + str(i) + '_of_' + str(n_shards) + '.hdf5' for i in range(n_shards)]

keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids', 'next_sentence_labels']

n_samples = int(0)
n_real_tokens = int(0)
n_real_mask = int(0)

alloc_size = 77000 * 2048  # A slight overestimate of n_samples

real_tokens = np.zeros((alloc_size), dtype=np.uint64)
real_mask = np.zeros((alloc_size), dtype=np.uint64)

idx = int(0)
for input_file in input_files:
  print(input_file)
  hfile = h5py.File(input_file, 'r')
  
  inputs = [np.asarray(hfile[key][:]) for key in keys]
  n_samples_shard = int(inputs[5][:].shape[0])
  n_tokens_per_seq = int(inputs[0].shape[1])
  n_mask_per_seq = int(inputs[3].shape[1])

  for i in range(n_samples_shard):
    curr_real_tokens = np.sum(inputs[1][i,:] > 0)
    curr_real_mask = np.sum(inputs[3][i, :] > 0)
    real_tokens[idx] = curr_real_tokens
    real_mask[idx] = curr_real_mask
    idx += 1
  
  n_samples += n_samples_shard

  hfile.close()

print('n_samples:,', n_samples)
print('n_tokens_per_seq:', n_tokens_per_seq)
print('n_mask_per_seq:', n_mask_per_seq)
print('total n_pad_tokens:', np.sum(n_tokens_per_seq - real_tokens[:n_samples]))
print('total n_pad_mask_tokens:', np.sum(n_mask_per_seq - real_mask[:n_samples]))
print('mean pad tokens per seq:', np.mean(real_tokens[:n_samples]))
print('mean pad masks per seq:', np.mean(real_mask[:n_samples]))

