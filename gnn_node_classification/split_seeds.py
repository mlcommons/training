import argparse
import os.path as osp
import torch

class SeedSplitter(object):
  def __init__(self,
               path,
               dataset_size='tiny',
               use_label_2K=True,
               random_seed=42,
               validation_frac=0.01):
    self.path = path
    self.dataset_size = dataset_size
    self.use_label_2K = use_label_2K
    self.random_seed = random_seed
    self.validation_frac = validation_frac
    self.paper_nodes_num = {'tiny':100000, 'small':1000000, 'medium':10000000, 'large':100000000, 'full':269346174}
    self.process()
  
  def process(self):
    torch.manual_seed(self.random_seed)
    n_labeled_idx = self.paper_nodes_num[self.dataset_size]
    if self.dataset_size == 'full':
      if self.use_label_2K:
          n_labeled_idx = 157675969
      else:
          n_labeled_idx = 227130858

    shuffled_index = torch.randperm(n_labeled_idx)
    n_train = int(n_labeled_idx * 0.6)
    n_val = int(n_labeled_idx * self.validation_frac)

    train_idx = shuffled_index[:n_train]
    val_idx = shuffled_index[n_train : n_train + n_val]

    path = osp.join(self.path, self.dataset_size, 'processed')
    torch.save(train_idx, osp.join(path, 'train_idx.pt'))
    torch.save(val_idx, osp.join(path, 'val_idx.pt'))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  root = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__)))), 'data', 'igbh')
  parser.add_argument('--path', type=str, default=root,
      help='path containing the datasets')
  parser.add_argument('--dataset_size', type=str, default='full',
      choices=['tiny', 'small', 'medium', 'large', 'full'],
      help='size of the datasets')
  parser.add_argument("--random_seed", type=int, default='42')
  parser.add_argument('--num_classes', type=int, default=2983,
      choices=[19, 2983], help='number of classes')
  parser.add_argument("--validation_frac", type=float, default=0.005,
      help="Fraction of labeled vertices to be used for validation.")
  
  args = parser.parse_args()
  splitter = SeedSplitter(path=args.path,
                          dataset_size=args.dataset_size,
                          use_label_2K=(args.num_classes==2983),
                          random_seed=args.random_seed,
                          validation_frac=args.validation_frac)