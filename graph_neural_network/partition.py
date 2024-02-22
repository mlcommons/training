import argparse
import os.path as osp

import graphlearn_torch as glt
import torch

from dataset import IGBHeteroDataset
from typing import Literal

def partition_dataset(src_path: str,
                      dst_path: str,
                      num_partitions: int,
                      chunk_size: int,
                      dataset_size: str='tiny',
                      in_memory: bool=True,
                      edge_assign_strategy: str='by_src',
                      use_label_2K: bool=False,
                      with_feature: bool=True,
                      use_fp16: bool=False,
                      layout: Literal['CSC', 'CSR', 'COO'] = 'COO'):
  print(f'-- Loading igbh_{dataset_size} ...')
  data = IGBHeteroDataset(src_path, dataset_size, in_memory, use_label_2K, use_fp16=use_fp16)
  node_num = {k : v.shape[0] for k, v in data.feat_dict.items()}

  print('-- Saving label ...')
  label_dir = osp.join(dst_path, f'{dataset_size}-label')
  glt.utils.ensure_dir(label_dir)
  torch.save(data.label.squeeze(), osp.join(label_dir, 'label.pt'))

  print('-- Partitioning training idx ...')
  train_idx = data.train_idx
  train_idx = train_idx.split(train_idx.size(0) // num_partitions)
  train_idx_partitions_dir = osp.join(dst_path, f'{dataset_size}-train-partitions')
  glt.utils.ensure_dir(train_idx_partitions_dir)
  for pidx in range(num_partitions):
    torch.save(train_idx[pidx], osp.join(train_idx_partitions_dir, f'partition{pidx}.pt'))

  print('-- Partitioning validation idx ...')
  val_idx = data.val_idx
  val_idx = val_idx.split(val_idx.size(0) // num_partitions)
  val_idx_partitions_dir = osp.join(dst_path, f'{dataset_size}-val-partitions')
  glt.utils.ensure_dir(val_idx_partitions_dir)
  for pidx in range(num_partitions):
    torch.save(val_idx[pidx], osp.join(val_idx_partitions_dir, f'partition{pidx}.pt'))

  print('-- Partitioning graph and features ...')
  partitions_dir = osp.join(dst_path, f'{dataset_size}-partitions')
  partitioner = glt.partition.RandomPartitioner(
    output_dir=partitions_dir,
    num_parts=num_partitions,
    num_nodes=node_num,
    edge_index=data.edge_dict,
    node_feat=data.feat_dict,
    node_feat_dtype = torch.float16 if use_fp16 else torch.float32,
    edge_assign_strategy=edge_assign_strategy,
    chunk_size=chunk_size,
  )
  partitioner.partition(with_feature)

  if layout in ['CSC', 'CSR']:
    compress_edge_dict = {}
    compress_edge_dict[('paper', 'cites', 'paper')] = 'paper__cites__paper'
    compress_edge_dict[('paper', 'written_by', 'author')] = 'paper__written_by__author'
    compress_edge_dict[('author', 'affiliated_to', 'institute')] = 'author__affiliated_to__institute'
    compress_edge_dict[('paper', 'topic', 'fos')] = 'paper__topic__fos'
    compress_edge_dict[('author', 'rev_written_by', 'paper')] = 'author__rev_written_by__paper'
    compress_edge_dict[('institute', 'rev_affiliated_to', 'author')] = 'institute__rev_affiliated_to__author'
    compress_edge_dict[('fos', 'rev_topic', 'paper')] = 'fos__rev_topic__paper'
    compress_edge_dict[('paper', 'published', 'journal')] = 'paper__published__journal'
    compress_edge_dict[('paper', 'venue', 'conference')] = 'paper__venue__conference'
    compress_edge_dict[('journal', 'rev_published', 'paper')] = 'journal__rev_published__paper'
    compress_edge_dict[('conference', 'rev_venue', 'paper')] = 'conference__rev_venue__paper'

    for pidx in range(num_partitions):
      base_path = osp.join(dst_path, f'{dataset_size}-partitions', f'part{pidx}', 'graph')
      device = torch.device('cpu')
      graph_dict = {}
      for etype, e_path in compress_edge_dict.items():
        graph = glt.partition.base.load_graph_partition_data(osp.join(base_path, e_path), device)
        if graph != None:
          graph_dict[etype] = graph
      
      edge_dir = 'out' if layout == 'CSR' else 'in'
      dataset = glt.distributed.DistDataset(edge_dir=edge_dir)
      edge_index, edge_ids, edge_weights = {}, {}, {}
      for k, v in graph_dict.items():
        edge_index[k] = v.edge_index
        edge_ids[k] = v.eids
        edge_weights[k] = v.weights 
      # COO is the oroginal layout of raw igbh graph
      dataset.init_graph(edge_index, edge_ids, edge_weights, layout='COO',
        graph_mode='CPU', device=device)

      for etype in graph_dict:
        graph = dataset.get_graph(etype)
        indptr, indices, _ = graph.export_topology()
        path = osp.join(base_path, compress_edge_dict[etype])
        if layout == 'CSR':
          torch.save(indptr, osp.join(path, 'rows.pt'))
          torch.save(indices, osp.join(path, 'cols.pt'))
        else:
          torch.save(indptr, osp.join(path, 'cols.pt'))
          torch.save(indices, osp.join(path, 'rows.pt'))

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
  parser.add_argument('--num_classes', type=int, default=2983,
      choices=[19, 2983], help='number of classes')
  parser.add_argument('--in_memory', type=int, default=0,
      choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
  parser.add_argument("--num_partitions", type=int, default=2,
      help="Number of partitions")
  parser.add_argument("--chunk_size", type=int, default=10000,
      help="Chunk size for feature partitioning.")
  parser.add_argument("--edge_assign_strategy", type=str, default='by_src',
      help="edge assign strategy can be either 'by_src' or 'by_dst'")
  parser.add_argument('--with_feature', type=int, default=1,
      choices=[0, 1], help='0:do not partition feature, 1:partition feature')
  parser.add_argument('--use_fp16', action="store_true",
      help="save partitioned node/edge feature into fp16 format")
  parser.add_argument("--layout", type=str, default='COO', 
      help="layout of the partitioned graph: CSC, CSR, COO")

  args = parser.parse_args()

  partition_dataset(
    args.src_path,
    args.dst_path,
    num_partitions=args.num_partitions,
    chunk_size=args.chunk_size,
    dataset_size=args.dataset_size,
    in_memory=args.in_memory,
    edge_assign_strategy=args.edge_assign_strategy,
    use_label_2K=args.num_classes==2983,
    with_feature=args.with_feature==1,
    use_fp16=args.use_fp16,
    layout = args.layout
  )
