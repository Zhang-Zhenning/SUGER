import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch_geometric.nn import GCNConv, RGCNConv, global_sort_pool, global_add_pool
from torch_geometric.utils import dropout_adj
import torch_geometric
import pdb
import time
import numpy as np

# base GNN class
class GNN(torch.nn.Module):
    def __init__(self,dataset,gconv=GCNConv,latent_dim=[32,32,32,1],regression=False,adj_dropout=0.2,force_undirected=False):
        super(GNN,self).__init__()
        self.regression = regression
        self.adj_dropout = adj_dropout
        self.force_undirected = force_undirected
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features,latent_dim[0]))
        for i in range(len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i],latent_dim[i+1]))
        self.lin1 = Linear(sum(latent_dim),128)

        if self.regression:
            self.lin2 = Linear(128,1)
        else:
            self.lin2 = Linear(128,dataset.num_classes)
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    
    def forward(self,data):
        x,edge_index,batch = data.x,data.edge_index,data.batch
        
        if self.adj_dropout > 0:
            edge_index,edge_type = dropout_adj(edge_index,edge_type,p=self.adj_dropout,force_undirected=self.force_undirected,num_nodes=len(x),training=self.training)
        
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x,edge_index))
            concat_states.append(x)
        
        # information aggregation : by concatenate
        concat_states = torch.cat(concat_states,1)

        x = global_add_pool(concat_states,batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x,p=0.5,training=self.training)
        x = self.lin2(x)

        if self.regression:
            return x[:,0]
        else:
            return F.log_softmax(x,dim=-1)
    
    def __repr__(self):
        return self.__class__.__name__


class BasicModel(GNN):
    # The GNN model of Inductive Graph-based Matrix Completion.
    # Use RGCN convolution + center-nodes readout.
    def __init__(self, dataset, gconv=RGCNConv, latent_dim=[32, 32, 32, 32],
                 num_relations=3, num_bases=2, regression=True, adj_dropout=0.2,
                 force_undirected=False, side_features=False, n_side_features=0,
                 multiply_by=1):
        super(BasicModel, self).__init__(
            dataset, GCNConv, latent_dim, regression, adj_dropout, force_undirected
        )
        self.multiply_by = multiply_by
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(dataset.num_features,
                          latent_dim[0], num_relations, num_bases))
        for i in range(0, len(latent_dim)-1):
            self.convs.append(
                gconv(latent_dim[i], latent_dim[i+1], num_relations, num_bases))
        self.lin1 = Linear(2*sum(latent_dim), 128)
        self.side_features = side_features
        if side_features:
            self.lin1 = Linear(2*sum(latent_dim)+n_side_features, 128)

    def forward(self, data):
        start = time.time()
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index, edge_type, p=self.adj_dropout,
                force_undirected=self.force_undirected, num_nodes=len(x),
                training=self.training
            )
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index, edge_type))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)

        # we aggregate all the information by concatenation, temporarily.
        bundles = data.x[:, 0] == 1
        users = data.x[:, 1] == 1
        user_f = concat_states[users]
        bundle_f = concat_states[bundles]
        
        # check error
        if bundle_f.size()[0] == 0:
            
            return None

        x = torch.cat([user_f, bundle_f], 1)
        if self.side_features:
            x = torch.cat([x, data.u_feature, data.v_feature], 1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            # the sigmoid here ensure output range in 0-1
            return torch.sigmoid(x[:, 0] * self.multiply_by)
        else:
            return F.log_softmax(x, dim=-1)


