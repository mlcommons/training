from argparse import ArgumentParser
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import numpy_indexed as npi
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

    return parser.parse_args()


def generate_negatives(sampler, num_negatives, users):
    neg_users_items = np.empty([num_negatives], object)
    for i in range(num_negatives):
        negatives = np.array([users, sampler.sample_negatives(users)])
        neg_users_items[i] = negatives.transpose()
    return neg_users_items;


def main():
    args = parse_args()
    train_ratings = torch.LongTensor()
    test_ratings_chunk = [torch.LongTensor()] * args.user_scaling
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

    def group_by_user():
      """This could be accomplished by doing a group-by user, then yielding
         rows.  However, fast group by on very large ratings pairs is
         expensive. This generator implementation takes the iter_fn out of
         the critical path.
      """
      cur_buf = 0 # toggle the two buffers                   
      outs = [[],[]]
      clear_buf = [False, False]
      cur_user_id = train_ratings[0][0]
      for pair in train_ratings:
        if pair[0] == cur_user_id:

          if clear_buf[0]:
            outs[0] = []
            clear_buf[0] = False
          if clear_buf[1]:
            outs[1] = []
            clear_buf[1] = False

          outs[cur_buf].append(pair[1])
        else:

          if clear_buf[0]:
            outs[0] = []
            clear_buf[0] = False
          if clear_buf[1]:
            outs[1] = []
            clear_buf[1] = False

          clear_buf[cur_buf] = True
          cur_user_id = pair[0]
          cur_buf = 1 - cur_buf
          outs[cur_buf].append(pair[1])
          yield np.array(outs[1 - cur_buf])
      yield np.array(outs[cur_buf])

    sampler, pos_users, pos_items  = process_data(
        num_items=nb_items, min_items_per_user=1, iter_fn=group_by_user)
    assert len(pos_users) == train_ratings.shape[0], "Cardinality difference with original data and sample table data."

    fn_prefix = CACHE_FN.format(args.user_scaling, args.item_scaling)
    sampler_cache = fn_prefix + "cached_sampler.pkl"
    with open(sampler_cache, "wb") as f:
        pickle.dump([sampler, pos_users, pos_items, nb_items], f, pickle.HIGHEST_PROTOCOL)

    print(datetime.now(), 'Generating negative test samples...')

    test_negatives = [torch.LongTensor()] * args.user_scaling
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

