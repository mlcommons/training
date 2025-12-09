import argparse, datetime, os
import numpy as np
import torch
import os.path as osp

import graphlearn_torch as glt

from dataset import float2half
from download import download_dataset
from torch_geometric.utils import add_self_loops, remove_self_loops
from typing import Literal


class IGBHeteroDatasetCompress(object):
  def __init__(self,
               path,
               dataset_size,
               layout: Literal['CSC', 'CSR'] = 'CSC',):
    self.dir = path
    self.dataset_size = dataset_size
    self.layout = layout

    self.ntypes = ['paper', 'author', 'institute', 'fos']
    self.etypes = None
    self.edge_dict = {}
    self.paper_nodes_num = {'tiny':100000, 'small':1000000, 'medium':10000000, 'large':100000000, 'full':269346174}
    self.author_nodes_num = {'tiny':357041, 'small':1926066, 'medium':15544654, 'large':116959896, 'full':277220883}
    if not osp.exists(osp.join(path, self.dataset_size, 'processed')):
      download_dataset(path, 'heterogeneous', dataset_size)
    self.process()

  def process(self):
    paper_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
    'paper__cites__paper', 'edge_index.npy'))).t()
    author_paper_edges = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
    'paper__written_by__author', 'edge_index.npy'))).t()
    affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
    'author__affiliated_to__institute', 'edge_index.npy'))).t()
    paper_fos_edges = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
    'paper__topic__fos', 'edge_index.npy'))).t()
    if self.dataset_size in ['large', 'full']:
      paper_published_journal = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
      'paper__published__journal', 'edge_index.npy'))).t()
      paper_venue_conference = torch.from_numpy(np.load(osp.join(self.dir, self.dataset_size, 'processed',
      'paper__venue__conference', 'edge_index.npy'))).t()

    cites_edge = add_self_loops(remove_self_loops(paper_paper_edges)[0])[0]
    self.edge_dict = {
        ('paper', 'cites', 'paper'): (torch.cat([cites_edge[1, :], cites_edge[0, :]]), torch.cat([cites_edge[0, :], cites_edge[1, :]])),
        ('paper', 'written_by', 'author'): author_paper_edges,
        ('author', 'affiliated_to', 'institute'): affiliation_author_edges,
        ('paper', 'topic', 'fos'): paper_fos_edges,
        ('author', 'rev_written_by', 'paper'): (author_paper_edges[1, :], author_paper_edges[0, :]),
        ('institute', 'rev_affiliated_to', 'author'): (affiliation_author_edges[1, :], affiliation_author_edges[0, :]),
        ('fos', 'rev_topic', 'paper'): (paper_fos_edges[1, :], paper_fos_edges[0, :])
    }
    if self.dataset_size in ['large', 'full']:
      self.edge_dict[('paper', 'published', 'journal')] = paper_published_journal
      self.edge_dict[('paper', 'venue', 'conference')] = paper_venue_conference
      self.edge_dict[('journal', 'rev_published', 'paper')] = (paper_published_journal[1, :], paper_published_journal[0, :])
      self.edge_dict[('conference', 'rev_venue', 'paper')] = (paper_venue_conference[1, :], paper_venue_conference[0, :])
    self.etypes = list(self.edge_dict.keys())

    # init graphlearn_torch Dataset.
    edge_dir = 'out' if self.layout == 'CSR' else 'in'
    glt_dataset = glt.data.Dataset(edge_dir=edge_dir)
    glt_dataset.init_graph(
      edge_index=self.edge_dict,
      graph_mode='CPU',
    )

    # save the corresponding csr or csc file
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

    for etype in self.etypes:
      graph = glt_dataset.get_graph(etype)
      indptr, indices, _ = graph.export_topology()
      path = os.path.join(self.dir, self.dataset_size, 'processed', self.layout, compress_edge_dict[etype])
      if not os.path.exists(path):
        os.makedirs(path)
      torch.save(indptr, os.path.join(path, 'indptr.pt'))
      torch.save(indices, os.path.join(path, 'indices.pt'))
    path = os.path.join(self.dir, self.dataset_size, 'processed', self.layout)
    print(f"The {self.layout} graph has been persisted in path: {path}")



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))), 'data', 'igbh')
  glt.utils.ensure_dir(root)
  parser.add_argument('--path', type=str, default=root,
      help='path containing the datasets')
  parser.add_argument('--dataset_size', type=str, default='full',
      choices=['tiny', 'small', 'medium', 'large', 'full'],
      help='size of the datasets')
  parser.add_argument("--layout", type=str, default='CSC')
  parser.add_argument('--use_fp16', action="store_true",
    help="convert the node/edge feature into fp16 format")
  args = parser.parse_args()
  print(f"Start constructing the {args.layout} graph...")
  igbh_dataset = IGBHeteroDatasetCompress(args.path, args.dataset_size, args.layout)
  if args.use_fp16:
    base_path = osp.join(args.path, args.dataset_size, 'processed')
    float2half(base_path, args.dataset_size)
  



