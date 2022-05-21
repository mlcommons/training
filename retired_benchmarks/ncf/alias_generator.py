# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
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
# ==============================================================================
"""Perform negative sampling using alias tables.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import multiprocessing.dummy
import os
import pickle
import timeit

import numpy as np
import scipy.stats

try:
    from tensorflow.gfile import Open
except ImportError:
    # Remote file systems will not be available.
    Open = open

# Needed to load metadata file
# import graph_expansion  # pylint: disable=unused-import


_PREFIX = "16x32_"


# =============================================================================
# == Alias Table Computation ==================================================
# =============================================================================

# We compute the table in float64 so that we can perform very stringent
# floating point checks on the resulting alias table without having to worry
# about precision issues.
alias_compute_dtype = np.float64


def compute_alias_table(counts):
    # type: (np.ndarray) -> (np.ndarray, np.ndarray)
    """ Compute the alias table based on counts. (Unnormalized probability)

    For reference, see:
        http://www.keithschwarz.com/darts-dice-coins/
        http://www.keithschwarz.com/interesting/code/?dir=alias-method

    """

    # Initial bookkeeping
    num_regions = counts.shape[0]
    total_counts = np.sum(counts)

    # Compute the normalized probability.
    # NOTE: We will mutate p as we go. It is therefore not correct to use it later
    #       as the PDF of counts.
    p = counts.astype(alias_compute_dtype) / total_counts

    # Average (expectation) probability for each element.
    p_expect = np.ones((), dtype=alias_compute_dtype) / num_regions

    # Construct stacks and partition indices for our elements.
    low_stack, high_stack = [], []
    for i, pi in enumerate(p):
        if pi < p_expect:
            low_stack.append(i)
        else:
            high_stack.append(i)

    # Initialize the alias table
    alias_p = np.zeros(shape=num_regions, dtype=alias_compute_dtype)
    alias_ind = np.zeros(shape=num_regions, dtype=np.int32)

    while low_stack and high_stack:
        low_ind = low_stack.pop()
        p_low = p[low_ind]

        high_ind = high_stack.pop()
        p_high = p[high_ind]

        # The unused probability for the "low" index is aliased to the
        # high index.
        alias_p[low_ind] = p_low
        alias_ind[low_ind] = high_ind

        # We then decrement the aliased probability from the high index,
        # and push it to the appropriate stack based on the remaining p.
        p_remaining = p_high - (p_expect - p_low)
        p[high_ind] = p_remaining
        if p_remaining >= p_expect:
            high_stack.append(high_ind)
        else:
            low_stack.append(high_ind)

    # Once one stack is exhaused, the elements of the remaining stack are
    # guaranteed to have p_expect probability, so we assign that here.
    for ind in low_stack + high_stack:
        alias_p[ind] = p_expect
        alias_ind[ind] = ind

    # ===============================================================
    # == Testing ====================================================
    # ===============================================================
    #
    # If the alias table is wrong, it will be very hard to detect
    # later. As a result, it is worth it to perform two tests:
    #
    #   1) Check that we can recover p from the alias table. If so,
    #      then we have properly assigned our probabilities.
    #
    #   2) Make sure that we haven't accidentally overloaded the
    #      real / alias split.
    p_test = alias_p.copy()
    # Can't vectorize easily due to repeated indices in alias_ind
    for i, p in zip(alias_ind, alias_p):
        p_test[i] += p_expect - p

    assert np.allclose(p_test, counts / total_counts, atol=0, rtol=1e-8)
    alias_split_p = alias_p / p_expect
    assert np.all(alias_split_p <= 1. + 1e-8)

    return alias_ind, alias_split_p.astype(np.float32)


class AliasSample(object):
    """Helper class for generating negatives."""

    def __init__(self, offsets, num_regions, region_cardinality,
                 region_starts, alias_index, alias_split_p):
        self.offsets = offsets
        self.num_regions = num_regions
        self.region_cardinality = region_cardinality
        self.region_starts = region_starts
        self.alias_index = alias_index
        self.alias_split_p = alias_split_p

    @staticmethod
    def new_rand_gen():
        """Helper method for creating random state objects.

        When generating random numbers, NumPy grabs the GIL to ensure repeatability. (NumPy makes very strong guarantees
        about random numbers.) This is a problem when trying to thread, because if random generation is a major cost
        then all threads bottleneck there. In practice for alias table generation, going above 3-4 threads when using
        `np.random.random / np.random.randint` is counter-productive. However, since we don't care about that it is
        overwhelmingly worthwhile to create new (independent) random generators on the fly.
        """
        uint32_max = (1 << 32) - 1
        return np.random.RandomState(seed=np.random.randint(uint32_max))

    def slightly_biased_randint(self, high):
        return np.mod(
            self.new_rand_gen().randint(low=0, high=np.iinfo(np.uint64).max,
                              size=high.shape, dtype=np.uint64),
            high.astype(np.uint64)
        ).astype(np.int32)

    def sample_negatives(self, user_ind):
        sample_offsets = self.offsets[user_ind]

        # First select a bin from the alias table
        bins = self.slightly_biased_randint(self.num_regions[user_ind])

        # Then determine whether to use the aliased region
        bin_lookup_ind = sample_offsets + bins
        threshold = self.alias_split_p[bin_lookup_ind]
        use_alias = self.new_rand_gen().uniform(size=user_ind.shape) > threshold
        chosen_region = bins * (1 - use_alias) + self.alias_index[bin_lookup_ind] * use_alias

        # Finally, select uniformly from the chosen region
        chosen_lookup_ind = sample_offsets + chosen_region
        lookup_offsets = self.slightly_biased_randint(self.region_cardinality[chosen_lookup_ind])
        chosen_items = self.region_starts[chosen_lookup_ind] + lookup_offsets

        return chosen_items


def process_data(num_items, min_items_per_user, iter_fn):
    user_id = -1
    user_id2 = -1
    positive_users = []
    positive_items = []
    num_regions, region_cardinality, region_starts, alias_index, alias_split_p = [], [], [], [], []
    for user_items in iter_fn():
        user_id2 += 1
        if len(user_items) < min_items_per_user:
            continue
        user_id += 1
        user_items = np.sort(user_items)

        positive_users.append(np.ones_like(user_items) * user_id)
        positive_items.append(user_items)

        bounds = np.concatenate([[-1], user_items, [num_items]])

        neg_region_starts = bounds[:-1] + 1
        neg_region_cardinality = bounds[1:] - bounds[:-1] - 1

        # Handle contiguous positives
        if np.min(neg_region_cardinality) == 0:
            filter_ind = neg_region_cardinality > 0
            neg_region_starts = neg_region_starts[filter_ind]
            neg_region_cardinality = neg_region_cardinality[filter_ind]
        user_alias_index, user_alias_split_p = compute_alias_table(neg_region_cardinality)

        num_regions.append(len(user_alias_index))
        region_cardinality.append(neg_region_cardinality)
        region_starts.append(neg_region_starts)
        alias_index.append(user_alias_index)
        alias_split_p.append(user_alias_split_p)

        if user_id % 10000 == 0:
            print("user id {} processed".format(user_id))

    return AliasSample(
        offsets=np.cumsum([0] + num_regions, dtype=np.int32)[:-1],
        num_regions=np.array(num_regions),
        region_cardinality=np.concatenate(region_cardinality),
        region_starts=np.concatenate(region_starts),
        alias_index=np.concatenate(alias_index),
        alias_split_p=np.concatenate(alias_split_p),
    ), np.concatenate(positive_users), np.concatenate(positive_items)


def make_synthetic_data(num_users, num_items, approx_items_per_user):
    output = []
    for _ in range(num_users):
        user_num_items = np.random.randint(low=int(approx_items_per_user * 0.5), high=int(approx_items_per_user * 1.5))
        items = np.array(sorted(set(np.random.randint(low=0, high=num_items, size=user_num_items))))
        output.append(items)

    def iter_fn():
        for i in output:
            yield i

    return iter_fn


def profile_sampler(sampler, batch_size, num_batches, num_users):
    st = timeit.default_timer()
    test_user_inds = [np.random.randint(low=0, high=num_users, size=batch_size) for _ in range(num_batches)]
    print("id creation time: {:.2f} sec".format(timeit.default_timer() - st))

    st = timeit.default_timer()
    k = 20  # Prevent single threaded test from taking forever.
    [sampler.sample_negatives(i) for i in test_user_inds]
    print("Single threaded time: {:.2f} sec / 1B samples".format((timeit.default_timer() - st) / min([k, batch_size]) / num_batches * 1e9))

    for num_threads in [4, 8, 16, 32, 48, 64]:
      with multiprocessing.dummy.Pool(num_threads) as pool:
          st = timeit.default_timer()
          results = pool.map(sampler.sample_negatives, test_user_inds)
          print("Multi threaded ({} threads) time: {:.1f} sec / 1B samples"
                .format(num_threads, (timeit.default_timer() - st) / batch_size / num_batches * 1e9))

    return test_user_inds, results


def test_using_synthetic():
    print("Testing alias method with synthetic data.")
    np.random.seed(0)
    num_users = 50  # Pick a small number of users to get good statistics.
    num_items = 1000
    iter_fn = make_synthetic_data(num_users=num_users, num_items=num_items, approx_items_per_user=100)
    sampler, pos_users, pos_items = process_data(num_items=num_items, min_items_per_user=1, iter_fn=iter_fn)

    test_user_inds, results = profile_sampler(sampler=sampler, batch_size=int(1e5), num_batches=100, num_users=num_users)

    # check_results
    positive_set = set()
    neg_counts = [num_items for _ in range(num_users)]
    for user, item in zip(pos_users, pos_items):
        positive_set.add((user, item))
        neg_counts[user] -= 1

    neg_sample_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    for user_inds, result_items in zip(test_user_inds, results):
        for user, item in zip(user_inds, result_items):
            if (user, item) in positive_set:
                raise ValueError("Negative sampling returned a positive.")
            neg_sample_counts[user][item] += 1

    for user_id in range(num_users):
        counts = np.array(list(neg_sample_counts[user_id].values()))

        # We sample enough that we expect all negatives to appear at least once.
        assert len(counts) == neg_counts[user_id]

        # Use a chi-square test for goodness of fit against a uniform distribution
        observed_p = counts / np.sum(counts)
        expected_p = np.ones_like(counts) / len(counts)
        assert scipy.stats.chisquare(observed_p, expected_p).statistic < 0.01

    print("Smoke test using synthetic data was a success!\n")


def iter_data():
    shards = sorted([i for i in os.listdir(os.getcwd())
                     if i.startswith(_PREFIX + "_train.")])

    for shard in shards:
        print(shard)
        with open(shard, "rb") as f:
            for i, data in enumerate(pickle.load(f)):
                yield data


def run_real_data():
    print("Starting on real data.")
    metadata_path = "{}_train_metadata.pkl".format(_PREFIX)
    with Open(metadata_path, "rb") as f:
        train_metadata = pickle.load(f)
    num_items = train_metadata.num_cols
    print("num_items:", num_items)

    st = timeit.default_timer()
    sampler_cache = _PREFIX + "cached_sampler.pkl"
    if os.path.exists(sampler_cache):
      print("Using cache: {}".format(sampler_cache))
      with open(sampler_cache, "rb") as f:
        sampler, pos_users, pos_items = pickle.load(f)
    else:
      sampler, pos_users, pos_items = process_data(num_items=num_items, min_items_per_user=1, iter_fn=iter_data)
      with open(sampler_cache, "wb") as f:
        pickle.dump([sampler, pos_users, pos_items], f, pickle.HIGHEST_PROTOCOL)
    preproc_time = timeit.default_timer() - st
    num_users = len(sampler.num_regions)
    print("num_users:", num_users)
    print("Preprocessing complete: {:.1f} sec".format(preproc_time))
    print()

    _ = profile_sampler(sampler=sampler, batch_size=int(1e6), num_batches=1000, num_users=num_users)


def main():
    test_using_synthetic()
    run_real_data()


if __name__ == "__main__":
    main()
