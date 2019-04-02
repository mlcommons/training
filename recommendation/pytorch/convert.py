from argparse import ArgumentParser
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import numpy_indexed as npi
import pickle
from alias_generator import process_data
from alias_generator import profile_sampler

_PREFIX = "16x32_"

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


def generate_negatives(sampler, num_negatives, test_ratings):
    users = test_ratings[:,0].numpy()
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
        train_ratings = torch.cat((train_ratings,
            torch.from_numpy(np.load(args.data + '/trainx'
                + str(args.user_scaling) + 'x' + str(args.item_scaling)
                + '_' + str(chunk) + '.npz', encoding='bytes')['arr_0'])))
        test_ratings_chunk[chunk] = torch.from_numpy(np.load(args.data + '/testx'
                + str(args.user_scaling) + 'x' + str(args.item_scaling)
                + '_' + str(chunk) + '.npz', encoding='bytes')['arr_0'])
        test_chunk_size[chunk] = test_ratings_chunk[chunk].shape[0]

    nb_maxs = torch.max(train_ratings, 0)[0]
    nb_users = nb_maxs[0].item()+1
    nb_items = nb_maxs[1].item()+1
    print(datetime.now(), 'number of users, items...', nb_users, nb_items)
    train_input = npi.group_by(train_ratings[:, 0]).split(train_ratings[:, 1])

    def iter_fn():
      for _, items in enumerate(train_input):
         yield items

    sampler, pos_users, pos_items = process_data(num_items=nb_items, min_items_per_user=1, iter_fn=iter_fn)

    sampler_cache = _PREFIX + "cached_sampler.pkl"
    with open(sampler_cache, "wb") as f:
        pickle.dump([sampler, pos_users, pos_items], f, pickle.HIGHEST_PROTOCOL)

    print(datetime.now(), 'Generating negative test samples...')

    test_negatives = [torch.LongTensor()] * args.user_scaling
    test_user_offset = 0
    for chunk in range(args.user_scaling):
        test_negatives[chunk] = generate_negatives(
                sampler,
                args.valid_negative,
                test_ratings_chunk[chunk])

        file_name = (args.data + '/test_negx' + str(args.user_scaling) + 'x'
                + str(args.item_scaling) + '_' + str(chunk) + '.npz')

        print("test negatives:", len(test_negatives[chunk]))
        np.savez_compressed(file_name, test_negatives[chunk])
        print(datetime.now(), 'Chunk ', chunk, '/', args.user_scaling, 'saved.')
        test_user_offset += test_chunk_size[chunk]


if __name__ == '__main__':
    main()

