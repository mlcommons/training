from argparse import ArgumentParser
import numpy as np
import torch
from datetime import datetime
import numpy_indexed as npi
import os
import pickle
from alias_generator import process_data

CACHE_FN = "alias_tbl_{}x{}_"

def parse_args():
    parser = ArgumentParser(description="Train a Nerual Collaborative"
                                        " Filtering model converter")
    parser.add_argument('data', type=str,
                        help='path to test and training data files')
    parser.add_argument('--valid-negative', type=int, default=999,
                        help='Number of negative samples for each positive test example')
    parser.add_argument('--user_scaling', default=16, type=int)
    parser.add_argument('--item_scaling', default=32, type=int)
    parser.add_argument('--use_sampler_cache', action='store_true',
                        help='Use exiting pre-processed sampler cache. See CACHE_FN variable and use.')

    return parser.parse_args()


def generate_negatives(sampler, num_negatives, users):
    neg_users_items = np.empty([num_negatives], object)
    for i in range(num_negatives):
        negatives = np.array([users, sampler.sample_negatives(users)], dtype=np.int32)
        neg_users_items[i] = negatives.transpose()
    return neg_users_items;


def process_raw_data(args):
    train_ratings = torch.IntTensor()
    test_ratings_chunk = [torch.IntTensor()] * args.user_scaling
    #train_ratings = torch.LongTensor()
    #test_ratings_chunk = [torch.LongTensor()] * args.user_scaling
    test_chunk_size = [0] * args.user_scaling
    for chunk in range(args.user_scaling):
        print(datetime.now(), "Loading data chunk {} of {}".format(chunk+1, args.user_scaling))
        train_ratings = torch.cat((train_ratings,
            torch.from_numpy(np.load(args.data + '/trainx'
                + str(args.user_scaling) + 'x' + str(args.item_scaling)
                + '_' + str(chunk) + '.npz', encoding='bytes')['arr_0'])))
        test_ratings_chunk[chunk] = torch.from_numpy(np.load(args.data + '/testx'
                + str(args.user_scaling) + 'x' + str(args.item_scaling)
                + '_' + str(chunk) + '.npz', encoding='bytes')['arr_0'])
        test_chunk_size[chunk] = test_ratings_chunk[chunk].shape[0]

    # Due to the fractal graph expansion process, some generated users do not
    # have any ratings. Therefore, nb_users should not be max_user_index+1.
    nb_users = len(np.unique(train_ratings[:, 0]))

    # nb_items should remain as max_item_index+1 -- users can rate any movie
    nb_maxs = torch.max(train_ratings, 0)[0]
    nb_items = nb_maxs[1].item()+1  # Zero is valid item in output from expansion
    del nb_maxs
    print(datetime.now(), "Number of users: {}, Number of items: {}".format(nb_users, nb_items))
    print(datetime.now(), "Number of ratings: {}".format(train_ratings.shape[0]))

    train_input = npi.group_by(train_ratings[:, 0]).split(train_ratings[:, 1])
    def iter_fn_simple():
      for _, items in enumerate(train_input):
         yield items

    sampler, pos_users, pos_items  = process_data(
        num_items=nb_items, min_items_per_user=1, iter_fn=iter_fn_simple)
    assert len(pos_users) == train_ratings.shape[0], "Cardinality difference with original data and sample table data."

    print("pos_users type: {}, pos_items type: {}, s.offsets: {}".format(
          pos_users.dtype, pos_items.dtype, sampler.offsets.dtype))
    print("num_reg: {}, region_card: {}".format(sampler.num_regions.dtype,
          sampler.region_cardinality.dtype))
    print("region_starts: {}, alias_index: {}, alias_p: {}".format(
          sampler.region_starts.dtype, sampler.alias_index.dtype,
          sampler.alias_split_p.dtype))

    fn_prefix = CACHE_FN.format(args.user_scaling, args.item_scaling)
    sampler_cache = fn_prefix + "cached_sampler.pkl"
    with open(sampler_cache, "wb") as f:
        pickle.dump([sampler, pos_users, pos_items, nb_items, test_chunk_size], f, pickle.HIGHEST_PROTOCOL)

    return sampler, test_chunk_size

def main():
    args = parse_args()

    if not args.use_sampler_cache:
      sampler, test_chunk_size = process_raw_data(args)
    else:
      fn_prefix = CACHE_FN.format(args.user_scaling, args.item_scaling)
      sampler_cache = fn_prefix + "cached_sampler.pkl"
      print(datetime.now(), "Loading preprocessed sampler.")
      if os.path.exists(args.data):
        print("Using alias file: {}".format(args.data))
        with open(sampler_cache, "rb") as f:
          sampler, pos_users, pos_items, nb_items, test_chunk_size = pickle.load(f)

    print(datetime.now(), 'Generating negative test samples...')

## Test the 64-bit model while using CPU dataloader. Perhaps it is not OOM

    test_negatives = [torch.IntTensor()] * args.user_scaling
    test_user_offset = 0
    for chunk in range(args.user_scaling):
        neg_users = np.arange(test_user_offset,
            test_user_offset+test_chunk_size[chunk])
        test_negatives[chunk] = generate_negatives(
                sampler,
                args.valid_negative,
                neg_users)
        file_name = (args.data + '/test_negx' + str(args.user_scaling) + 'x'
                + str(args.item_scaling) + '_' + str(chunk) + '.npz')

        np.savez_compressed(file_name, test_negatives[chunk])
        print(datetime.now(), 'Chunk', chunk+1, 'of', args.user_scaling, 'saved.')
        test_user_offset += test_chunk_size[chunk]

    print(datetime.now(), "Number of test users: {}".format(test_user_offset))


if __name__ == '__main__':
    main()

