import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from queue import Queue
from random import shuffle, seed
import sys
import pickle

VOCAB_FILE = "vocab.ende.32768"

class TextDataset(Dataset):
  def __init__(self, data_dir, total_shards,
                 max_length, batch_size, mode="train"):
    self.data_dir = data_dir
    self.total_shards = total_shards
    self.max_length = max_length
    self.batch_size = batch_size
    self.mode = mode
    self.initialize_params()

  def initialize_params(self):
    """ Initializes dataset parameters and creates buckets and min_max boundaries
    """
    if self.mode == "train":
        self.prefix = "wmt32k-train-"
    elif self.mode == "eval":
        self.prefix = "wmt32k-dev-"
    self.suffix = "-of-" + str("{:05}".format(self.total_shards)) + ".pkl"

    self._BOUNDARY_SCALE = 1.1
    self._MIN_BOUNDARY = 8
    self._create_min_max_boundaries()
    self.bucket_batch_sizes = [int(self.batch_size / x.float()) for x in self.buckets_max]

    self.shard_q = Queue()
    self.batches = [[] for batch in range(self.total_buckets)]
    # Initialize bid as total_buckets so that multiple update_data aren't called
    self.bid = self.total_buckets

    seed(torch.initial_seed())
    self.rand_to_orig = list(range(self.total_buckets))

  def _get_example_length(self, example):
    """Returns the maximum length between the example inputs and targets."""
    length = max(example[0].shape[0], example[1].shape[0])
    return length

  def _exceeds_length(self, example_len, max_length=256):
    """Indicates whether the example's length is lower than the maximum length."""
    return example_len > max_length


  def _create_min_max_boundaries(self):
    """Create min and max boundary lists up to max_length.
    For example, when max_length=24, min_boundary=4 and boundary_scale=2, the
    returned values will be:
    buckets_min = [0, 4, 8, 16, 24]
    buckets_max = [4, 8, 16, 24, 25]

    Args:
    max_length: The maximum length of example in dataset.
    min_boundary: Minimum length in boundary.
    boundary_scale: Amount to scale consecutive boundaries in the list.

    Returns:
    min and max boundary lists

    """
    # Create bucket boundaries list by scaling the previous boundary or adding 1
    # (to ensure increasing boundary sizes).
    bucket_boundaries = []
    x = self._MIN_BOUNDARY
    while x < self.max_length:
      bucket_boundaries.append(x)
      x = max(x + 1, int(x * self._BOUNDARY_SCALE))

    # Create min and max boundary lists from the initial list.
    self.buckets_min = torch.tensor([0] + bucket_boundaries)
    self.buckets_max = torch.tensor(bucket_boundaries + [self.max_length + 1])
    self.total_buckets = int(self.buckets_max.shape[0])

  def window_size_fn(self, bucket_id):
    """Return number of examples to be grouped when given a bucket id."""
    return self.bucket_batch_sizes[bucket_id]

  def pad_example(self, example, padding):
    """Pads a one dimensional tensor"""
    return F.pad(example, padding)


  def example_to_bucket_id(self, example):
    """Return int64 bucket id for this example, calculated based on length."""

    seq_length = self._get_example_length(example)
    conditions_c = (self.buckets_min <= seq_length) * (seq_length < self.buckets_max)
    bucket_id = conditions_c.nonzero().min()
    return bucket_id

  def load_from_file(self, filepath):
    """ Loads binary data file using torch.load() API
    which uses pickle for (de)serializing.
    """
    examples = []
    with open(filepath, "rb") as file:
      while True:
        try:
          example = pickle.load(file)
          example = (torch.tensor(example[0]).float(), torch.tensor(example[1]).float())
          examples.append(example)
        except EOFError:
          break
    return examples

  def read_new_shard(self):
    """ Reads a new shard and updates self.batches and self.buckets
    """
    shard_num = self.shard_q.get()
    filepath = self.data_dir + self.prefix + "{:05}".format(shard_num) + self.suffix
    examples = self.load_from_file(filepath)

    # Reset self.batches_padding
    self.batches_padding = [0 for batch in range(self.total_buckets)]

    for example in examples:
      seq_length = self._get_example_length(example)
      if self._exceeds_length(seq_length, self.max_length):
        continue
      bid = self.example_to_bucket_id(example)

      self.batches[bid].append(example)
      if seq_length > self.batches_padding[bid]:
        self.batches_padding[bid] = seq_length

    # Randomize the batches
    shuffle(self.rand_to_orig)

  def pad_batch(self, batch):
    """ Pads an input batch to make dimensions of max_len
    Args:
      max_len: maximum length to pad for
      batch: Input batch in form of [inputs, targets]
    Returns:
      Padded batch of inputs with same form of [inputs, targets]
    """
    max_len = self.batches_padding[self.rand_to_orig[self.bid]]
    padded_batch = []
    for example in batch:
      # Padding is done to the right
      input_padding = [0, max_len - example[0].shape[0]]
      target_padding = [0, max_len - example[1].shape[0]]
      padded_example = (self.pad_example(example[0], input_padding),
                        self.pad_example(example[1], target_padding))
      padded_batch.append(padded_example)
    return padded_batch

  def get_next_batch(self):
    """ Gets batch of data for inputs having same batch id
    upto the maximum size of window_size(self.bid)
    Returns:
      Batch of inputs in form of [inputs, targets]
    """
    current_batchsize = 0
    max_batchsize = self.window_size_fn(self.rand_to_orig[self.bid])
    batch = []
    while (len(self.batches[self.rand_to_orig[self.bid]]) > 0 and
          current_batchsize < max_batchsize):
      example = self.batches[self.rand_to_orig[self.bid]].pop()
      batch.append(example)
      current_batchsize += 1
    return batch

  def get_padded_batch(self):
    batch = self.get_next_batch()
    padded_batch = self.pad_batch(batch)
    padded_batch = padded_batch = (torch.stack(list(zip(*padded_batch))[0]),
                                   torch.stack(list(zip(*padded_batch))[1]))
    return padded_batch

  def update_data(self):
    """ Updates self.batches, indexes and batch_id and
    read new shard if required
    """
    # Checks if all batches for current shard has been read
    if self.bid == len(self.batches):
      """ Checks if all shards assigned to self have been
      consumed. If not then reads new shard.
      """
      if self.shard_q.empty():
        raise StopIteration
      self.read_new_shard()
      self.bid = 0

    while len(self.batches[self.rand_to_orig[self.bid]]) == 0:
      self.bid += 1
      self.update_data()
    
  def __len__(self):
    # Returns a very high number since the actual length is unknown
    return sys.maxsize

  def __getitem__(self, shard_num):
    shard_num = shard_num + 1
    if shard_num <= self.total_shards:
      self.shard_q.put(shard_num)
    self.update_data()
    padded_batch = self.get_padded_batch()
    return padded_batch
