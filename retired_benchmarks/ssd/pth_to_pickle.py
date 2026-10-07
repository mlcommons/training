#!/usr/bin/env python3

from argparse import ArgumentParser
import pickle
import numpy as np
import torch

parser = ArgumentParser(description="Convert a pytorch (.pth) file to a pickled dictionary of numpy arrays. "
                                    "The dictionary will have the following format: \n"
                                    "{pytorch param name: numpy array}")
parser.add_argument('input_file', type=str, help='input pytorch .pth file')
parser.add_argument('output_file', type=str, help='output pickle file')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='print parameters names and statistics')
args = parser.parse_args()

dict_out = {}
pth_input = torch.load(open(args.input_file, 'rb'))

for key, value in pth_input.items():
    dict_out[key] = value.data.numpy()

if args.verbose:
    print("name, dtype, mean, std, min, max")
    for key, value in dict_out.items():
        t_mean = np.mean(value)
        t_std = np.std(value)
        t_min = np.min(value)
        t_max = np.max(value)
        print(f"{key}, {value.dtype}, {value.shape}, {t_mean:0.3}, {t_std:0.3}, {t_min:0.3}, {t_max:0.3}")

pickle.dump(dict_out, open(args.output_file, 'wb'))
