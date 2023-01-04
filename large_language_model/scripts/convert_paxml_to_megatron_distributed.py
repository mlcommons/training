# coding=utf-8
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import json
import multiprocessing
import os
import sys
from contextlib import nullcontext
from datetime import datetime
from time import time

import numpy as np
import tensorstore as ts
import torch
import zarr

INIT_TIME = time()


def get_ts_array(filename):
    spec = {'driver': 'zarr', 'metadata_key': '.zarray', 'kvstore': {}}
    spec['kvstore'] = {
        'driver': 'file',
        'path': filename,
    }

    return ts.open(ts.Spec(spec), open=True).result()


def _get_numpy_array(filename, layer_idx):
    t = get_ts_array(filename)
    if layer_idx is None:
        t_v = t.read().result()
    else:
        t_v = t[layer_idx:layer_idx + 1].read().result()
    return t_v


def get_array(filename, layer_idx):
    return _get_numpy_array(filename, layer_idx)


def store_array(np_arr, output_dir, lyr_name, layer_idx, layers_num, dtype):
    if dtype == 'bf16':
        dtype = np.dtype('bfloat16')
    elif dtype == 'fp32':
        dtype = np.float32
    else:
        raise NotImplementedError(f'Unsupported dtype {dtype}')
    np_arr = np_arr.astype(dtype)

    shape = list(np_arr.shape)
    if layer_idx is not None:
        shape[0] *= layers_num

    arr = open_array(lyr_name, shape, np_arr.dtype, output_dir)

    if layer_idx is None:
        arr.write(np_arr).result()
    else:
        arr[layer_idx:layer_idx + 1].write(np_arr).result()


def open_array(lyr_name, shape, dtype, output_dir):
    # Chunking heuristic, this is just to speed up saving (and later reading)
    chunks = list(shape)

    def maybe_reduce_chunk_size(dim, by):
        if len(chunks) > dim and chunks[dim] % by == 0:
            chunks[dim] = chunks[dim] // by

    if shape[0] > 200:
        maybe_reduce_chunk_size(0, 8)
        maybe_reduce_chunk_size(1, 8)
    else:
        chunks[0] = 1
        maybe_reduce_chunk_size(1, 8)
        maybe_reduce_chunk_size(2, 8)

    print(f'Saving {shape} array as {lyr_name} using chunks: {chunks}')

    fpath = os.path.join(output_dir, lyr_name)
    if os.path.exists(os.path.join(fpath, '.zarray')):
        create = False
        lock = nullcontext()
    else:
        create = True
        lock = zarr.ProcessSynchronizer(output_dir)[lyr_name + '.lock']

    with lock:
        if create:
            if os.path.exists(fpath):
                create = False
            else:
                os.makedirs(fpath)
        spec = {
            'driver': 'zarr',
            'metadata_key': '.zarray',
            'kvstore': {
                'driver': 'file',
                'path': fpath,
            },
            'create': create,
            'open': not create,
        }
        if create:
            spec['metadata'] = {
                'compressor': None,
                'dtype': 'bfloat16' if dtype == np.dtype('bfloat16') else '<f4',
                'shape': shape,
                'chunks': chunks,
            }
        arr = ts.open(spec).result()
    if create:
        os.remove(os.path.join(output_dir, lyr_name + '.lock'))
    return arr


def convert_layer(args, nv_name, g_name, nv_prefixes, dtypes, layer_idx):
    layer_init_time = time()
    print(F"G Name: {g_name}, NV name: {nv_name}, layer {layer_idx}")
    g_name_path = os.path.join(args.google_ckpts, g_name)
    array = get_array(g_name_path, layer_idx)
    print(F"G Name: {g_name}, NV name: {nv_name}, layer {layer_idx} read after {(time() - layer_init_time) / 60} minutes")
    if nv_name == "embedding.position_embeddings.weight":
        assert array.ndim == 2, array.ndim
        start_idx = 0
        end_idx = 2048
        array = array[start_idx: end_idx, :]
    elif nv_name == "embedding.word_embeddings.weight":
        assert array.ndim == 2, array.ndim
        array = array.transpose(1, 0)  # torch.transpose(0, 1)
    elif nv_name.startswith('encoder.final_layernorm'):
        array = array
    elif (
        nv_name.endswith("mlp.dense_4h_to_h.bias")
        or nv_name.endswith("post_attention_layernorm.bias")
        or nv_name.endswith("post_attention_layernorm.weight")
        or nv_name.endswith("input_layernorm.bias")
        or nv_name.endswith("input_layernorm.weight")
        or nv_name.endswith("self_attention.dense.bias")
        or nv_name.endswith("mlp.dense_h_to_4h.bias")
    ):
        array = array
    elif nv_name.endswith("self_attention.dense.weight"):
        assert array.ndim == 4, array.ndim
        array = array.reshape(array.shape[0], array.shape[1], -1)
    elif (
        nv_name.endswith("mlp.dense_h_to_4h.weight")
        or nv_name.endswith("mlp.dense_4h_to_h.weight")
    ):
        assert array.ndim == 3, array.ndim
        array = torch.from_numpy(array)
        array = array.permute(0, 2, 1)  # same as torch.transpose(0, 1) with extra prepended layer dimension
        array = array.numpy()
    elif nv_name.endswith("self_attention.query_key_value.weight"):
        assert array.ndim == 5, array.ndim
        # nv shape [4608, 12288] => 4608 = 12 (heads) * 3 (qkv) * 128 (hidden_size / heads)
        # google shape [96, 3, 12288, 96, 128]
        array = torch.from_numpy(array)  # Numpy can't handle that large arrays
        array = array.permute(0, 3, 1, 4, 2)  # same as torch.permute(2, 0, 3, 1) with extra prepended layer dimension
        array = array.reshape(array.shape[0], -1, array.shape[-1])
        array = array.numpy()
    elif nv_name.endswith("self_attention.query_key_value.bias"):
        assert array.ndim == 4, array.ndim
        # nv shape [4608] => 4608 = 12 (heads) * 3 (qkv) * 128 (hidden_size / heads)
        # google shape [96, 3, 96, 128]
        array = array.transpose(0, 2, 1, 3)  # same as torch.permute(1, 0, 2) with extra prepended layer dimension
        array = array.reshape(array.shape[0], -1)
    else:
        print(F"Not a valid layer name: {nv_name}")
        sys.exit(1)

    print(F"G Name: {g_name}, NV name: {nv_name}, layer {layer_idx} transformed after {(time() - layer_init_time) / 60} minutes")
    for nv_prefix, dtype in zip(nv_prefixes, dtypes):
        store_array(array, args.output_dir, nv_prefix + nv_name, layer_idx, args.num_layers, dtype)

    del array

    print(F"G Name: {g_name}, NV name: {nv_name}, layer {layer_idx} finished after {(time() - layer_init_time) / 60} minutes")



def arrange_google_ckpts(args):
    nv_g_names_pairs = [
            ("embedding.word_embeddings.weight", "params.lm.softmax.logits_ffn.linear.w"),
            ("embedding.position_embeddings.weight", "params.lm.position_emb.emb_var"),
            ("encoder.final_layernorm.weight", "params.lm.final_ln.scale"),
            ("encoder.final_layernorm.bias", "params.lm.final_ln.bias"),
    ]
    nv_g_names_repeat_pairs = [
            ("encoder.layers.post_attention_layernorm.bias", "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.bias"),
            ("encoder.layers.post_attention_layernorm.weight", "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.scale"),
            ("encoder.layers.input_layernorm.bias", "params.lm.transformer.repeat.sub.x_layers_0.layer_norm.bias"),
            ("encoder.layers.input_layernorm.weight", "params.lm.transformer.repeat.sub.x_layers_0.layer_norm.scale"),
            ("encoder.layers.mlp.dense_h_to_4h.bias", "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.bias.b"),
            ("encoder.layers.mlp.dense_h_to_4h.weight", "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.linear.w"),
            ("encoder.layers.mlp.dense_4h_to_h.weight", "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.linear.w"),
            ("encoder.layers.mlp.dense_4h_to_h.bias", "params.lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.bias.b"),
            ("encoder.layers.self_attention.dense.weight", "params.lm.transformer.repeat.sub.x_layers_0.self_attention.post.w"),
            ("encoder.layers.self_attention.dense.bias", "params.lm.transformer.repeat.sub.x_layers_0.self_attention.post.b"),
            ("encoder.layers.self_attention.query_key_value.weight", "params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.w"),
            ("encoder.layers.self_attention.query_key_value.bias", "params.lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.b"),
        ]

    nv_g_prefixes_dtype = [
        (['language_model.'], 'mdl_vars.', [args.dtype]),
        (['optimizer.state.exp_avg.language_model.'], 'opt_states_0.no_prefix_2.m.', ['fp32']),
        (['optimizer.state.exp_avg_sq.language_model.'], 'opt_states_0.no_prefix_2.v.', ['fp32']),
    ]

    nv_g_prefixes_repeat_dtype = [
        (['language_model.'], 'mdl_vars.', [args.dtype]),
        (['optimizer.state.exp_avg.language_model.'], 'opt_states_0.p#96#i-1_2.m.', ['fp32']),
        (['optimizer.state.exp_avg_sq.language_model.'], 'opt_states_0.p#96#i-1_2.v.', ['fp32']),
    ]

    if args.dtype == 'bf16':
        nv_g_prefixes_dtype[0][0].append('optimizer.state.fp32_from_fp16.language_model.')
        nv_g_prefixes_dtype[0][2].append('fp32')
        nv_g_prefixes_repeat_dtype[0][0].append('optimizer.state.fp32_from_fp16.language_model.')
        nv_g_prefixes_repeat_dtype[0][2].append('fp32')

    if args.pool == 0:
        process_parallel_in_groups(args, nv_g_names_pairs, nv_g_prefixes_dtype,
                                   nv_g_names_repeat_pairs, nv_g_prefixes_repeat_dtype)
        return

    convert_args = [
        (
            args,
            nv_name,
            g_prefix + g_name,
            nv_prefix,
            dtype,
            None
        )
        for nv_prefix, g_prefix, dtype in nv_g_prefixes_dtype
        for nv_name, g_name in nv_g_names_pairs
    ] + [
        (
            args,
            nv_name,
            g_prefix + g_name,
            nv_prefix,
            dtype,
            None
        )
        for nv_name, g_name in nv_g_names_repeat_pairs
        for nv_prefix, g_prefix, dtype in nv_g_prefixes_repeat_dtype
        # for layer_idx in range(args.num_layers)
    ]
    if args.pool == 1:
        list(map(convert_layer_early_error, convert_args))
    else:
        with multiprocessing.Pool(args.pool) as pool:
            list(pool.imap_unordered(
                convert_layer_early_error,
                convert_args,
            ))
        pool.join()


def convert_layer_early_error(convert_args):
    """
    Helper function to report errors right after they happen
    (not when joining the pool after the whole program ends).
    """
    try:
        return convert_layer(*convert_args)
    except Exception as e:
        print(f'ERROR occured while processing {convert_args[1:]}: {e}')
        raise e


def process_parallel_in_groups(args, nv_g_names_pairs, nv_g_prefixes_dtype,
                               nv_g_names_repeat_pairs, nv_g_prefixes_repeat_dtype):
    """ Process layers in groups (model params, optim first, second momentum) to avoid OOM. """
    pool_size = len(nv_g_names_pairs) + len(nv_g_names_repeat_pairs)
    print(f'Specified pool size 0, using pool size of {pool_size}')
    with multiprocessing.Pool(pool_size) as pool:
        assert len(nv_g_prefixes_dtype) == len(nv_g_prefixes_repeat_dtype)
        for layers_group_idx in range(len(nv_g_prefixes_dtype)):
            convert_args = []
            nv_prefix, g_prefix, dtype = nv_g_prefixes_dtype[layers_group_idx]
            convert_args.extend([
                (
                    args,
                    nv_name,
                    g_prefix + g_name,
                    nv_prefix,
                    dtype,
                    None
                )
                for nv_name, g_name in nv_g_names_pairs
            ])

            nv_prefix, g_prefix, dtype = nv_g_prefixes_repeat_dtype[layers_group_idx]
            convert_args.extend([
                (
                    args,
                    nv_name,
                    g_prefix + g_name,
                    nv_prefix,
                    dtype,
                    None
                )
                # for layer_idx in range(args.num_layers)
                for nv_name, g_name in nv_g_names_repeat_pairs
            ])

            list(pool.imap_unordered(
                convert_layer_early_error,
                convert_args,
            ))

            print(f'Finished processing layers group {layers_group_idx + 1}/{len(nv_g_prefixes_dtype)}')
            print('___')

    pool.join()


def add_metadata_file(output_dir):
    meta_fpath = os.path.join(output_dir, 'metadata.json')
    metadata = {'sharded_backend': 'zarr'}
    with open(meta_fpath, 'w') as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--google_ckpts', "-gckpt",
        type=str,
        help='Google Checkpoint directory')
    parser.add_argument(
        '--output_dir', "-o",
        type=str,
        help='Output directory (must exist)')
    parser.add_argument(
        '--dtype', "-dt",
        type=str,
        default="bf16",
        choices=['bf16', 'fp32'],
        help='Model data type (optimizer will be saved in fp32 anyway)')
    parser.add_argument(
        '--pool', "-p",
        type=int,
        default=0,
        help='Parallel processes. 0 means processing in groups'
             ' with pool size equal to number of layers')
    parser.add_argument(
        '--num_layers',
        type=int,
        default=96,
        help='Num model layers')
    parser.add_argument(
        '--no_add_metadata',
        action='store_false',
        dest='add_metadata',
        help='If True (default), creates metadata.json (part of unified checkpointing format)')

    args = parser.parse_args()
    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    start_time = datetime.now()
    if args.add_metadata:
        add_metadata_file(args.output_dir)
    arrange_google_ckpts(args)
    stop_time = datetime.now()
    run_time = stop_time - start_time
    print(f"[INFO] Spend {run_time} (h:m:s) to convert the model")
