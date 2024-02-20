import torch
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, GATConv, GCNConv, SAGEConv
from torch_geometric.utils import trim_to_layer

class RGNN(torch.nn.Module):
  r""" [Relational GNN model](https://arxiv.org/abs/1703.06103).

  Args:
    etypes: edge types.
    in_dim: input size.
    h_dim: Dimension of hidden layer.
    out_dim: Output dimension.
    num_layers: Number of conv layers.
    dropout: Dropout probability for hidden layers.
    model: "rsage" or "rgat".
    heads: Number of multi-head-attentions for GAT.
    node_type: The predict node type for node classification.

  """
  def __init__(self, etypes, in_dim, h_dim, out_dim, num_layers=2,
               dropout=0.2, model='rgat', heads=4, node_type=None, with_trim=False):
    super().__init__()
    self.node_type = node_type
    if node_type is not None:
      self.lin = torch.nn.Linear(h_dim, out_dim)

    self.convs = torch.nn.ModuleList()
    for i in range(num_layers):
      in_dim = in_dim if i == 0 else h_dim
      h_dim = out_dim if (i == (num_layers - 1) and node_type is None) else h_dim
      if model == 'rsage':
        self.convs.append(HeteroConv({
            etype: SAGEConv(in_dim, h_dim, root_weight=False)
            for etype in etypes}))
      elif model == 'rgat':
        self.convs.append(HeteroConv({
            etype: GATConv(in_dim, h_dim // heads, heads=heads, add_self_loops=False)
            for etype in etypes}))
    self.dropout = torch.nn.Dropout(dropout)
    self.with_trim = with_trim

  def forward(self, x_dict, edge_index_dict, num_sampled_edges_dict=None,
              num_sampled_nodes_dict=None):
    for i, conv in enumerate(self.convs):
      if self.with_trim:
        x_dict, edge_index_dict, _ = trim_to_layer(
          layer=i,
          num_sampled_nodes_per_hop=num_sampled_nodes_dict,
          num_sampled_edges_per_hop=num_sampled_edges_dict,
          x=x_dict,
          edge_index=edge_index_dict
        )
      for key in list(edge_index_dict.keys()):
        if key[0] not in x_dict or key[-1] not in x_dict:
          del edge_index_dict[key]
          
      x_dict = conv(x_dict, edge_index_dict)
      if i != len(self.convs) - 1:
        x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}
    if hasattr(self, 'lin'): # for node classification
      return self.lin(x_dict[self.node_type])
    else:
      return x_dict
