from argparse import ArgumentParser
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from negative_sampling import NegativeSampler

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

def main():
    args = parse_args()
    
    train_ratings = torch.LongTensor()
    test_chunk_size = [0] * args.user_scaling
    for chunk in range(args.user_scaling):
        train_ratings = torch.cat((train_ratings, 
            torch.from_numpy(np.load(args.data + '/trainx' 
                + str(args.user_scaling) + 'x' + str(args.item_scaling) 
                + '_' + str(chunk) + '.npz', encoding='bytes')['arr_0'])))
        test_ratings_chunk = torch.from_numpy(np.load(args.data + '/testx' 
                + str(args.user_scaling) + 'x' + str(args.item_scaling) 
                + '_' + str(chunk) + '.npz', encoding='bytes')['arr_0'])
        test_chunk_size[chunk] = test_ratings_chunk.shape[0]

    nb_maxs = torch.max(train_ratings, 0)[0]
    nb_users = nb_maxs[0].item()+1
    nb_items = nb_maxs[1].item()+1

    sampler = NegativeSampler(train_ratings, nb_users, nb_items)
    
    print(datetime.now(), 'Generating negative test samples...')
    
    test_negatives = [torch.LongTensor()] * args.user_scaling
    test_user_offset = 0
    for chunk in range(args.user_scaling):
        test_negatives[chunk] = sampler.generate_test_part(
                args.valid_negative, 
                test_chunk_size[chunk],
                test_user_offset) 
        
        file_name = (args.data + '/test_negx' + str(args.user_scaling) + 'x'
                + str(args.item_scaling) + '_' + str(chunk) + '.npz')

        np.savez_compressed(file_name, test_negatives[chunk])
        print(datetime.now(), 'Chunk ', chunk, '/', args.user_scaling, 'saved.')
        test_user_offset += test_chunk_size[chunk]
    

if __name__ == '__main__':
    main()
