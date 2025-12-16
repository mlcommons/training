import argparse
import os.path as osp

import graphlearn_torch as glt
import torch

from dataset import IGBHeteroDataset


def partition_feature(src_path: str,
                      dst_path: str,
                      partition_idx: int,
                      chunk_size: int,
                      dataset_size: str='tiny',
                      in_memory: bool=True,
                      use_fp16: bool=False):
  print(f'-- Loading igbh_{dataset_size} ...')
  data = IGBHeteroDataset(src_path, dataset_size, in_memory, with_edges=False, use_fp16=use_fp16)

  print(f'-- Build feature for partition {partition_idx} ...')
  dst_path = osp.join(dst_path, f'{dataset_size}-partitions')
  node_feat_dtype = torch.float16 if use_fp16 else torch.float32
  glt.partition.base.build_partition_feature(root_dir = dst_path,
                                             partition_idx = partition_idx,
                                             chunk_size = chunk_size,
                                             node_feat = data.feat_dict,
                                             node_feat_dtype = node_feat_dtype)


if __name__ == '__main__':
  root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))), 'data', 'igbh')
  glt.utils.ensure_dir(root)
  parser = argparse.ArgumentParser(description="Arguments for partitioning ogbn datasets.")
  parser.add_argument('--src_path', type=str, default=root,
      help='path containing the datasets')
  parser.add_argument('--dst_path', type=str, default=root,
      help='path containing the partitioned datasets')
  parser.add_argument('--dataset_size', type=str, default='full',
      choices=['tiny', 'small', 'medium', 'large', 'full'],
      help='size of the datasets')
  parser.add_argument('--in_memory', type=int, default=0,
      choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
  parser.add_argument("--partition_idx", type=int, default=0,
      help="Index of a partition")
  parser.add_argument("--chunk_size", type=int, default=10000,
      help="Chunk size for feature partitioning.")
  parser.add_argument("--use_fp16", action="store_true",
      help="save node/edge feature using fp16 format")


  args = parser.parse_args()

  partition_feature(
    args.src_path,
    args.dst_path,
    partition_idx=args.partition_idx,
    chunk_size=args.chunk_size,
    dataset_size=args.dataset_size,
    in_memory=args.in_memory==1,
    use_fp16=args.use_fp16
  )
