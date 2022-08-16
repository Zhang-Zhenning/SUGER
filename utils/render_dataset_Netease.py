# import torch　
import pickle
import sys
import os
import numpy as np
from baseline_preprocess_Netease import *
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset, InMemoryDataset

sys.path.extend(
    [data_root])
from model import *


# now we have gotten the subgraphs, we need to packge them here to get trainable daset

# define my dataset
class MyDataset(InMemoryDataset):
    def __init__(self, root, datas):
        self.datas = datas
        super(MyDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        name = 'training_data.pt'
        return [name]

    def process(self):
        # Extract enclosing subgraphs and save to disk
        data_list = self.datas
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        del data_list


def get_x(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x


def construct_graph_data(sub_graph):
    # there are 5 kinds of nodes
    x_length = 5

    # get user-bundle pair
    user_global_id = sub_graph[0]
    bundle_global_id = sub_graph[1]

    if id_to_AttNum(user_global_id)[0] == "bundle":
        user_global_id = sub_graph[1]
        bundle_global_id = sub_graph[0]

    _, user_private_id = id_to_AttNum(user_global_id)
    _, bundle_private_id = id_to_AttNum(bundle_global_id)

    # get nodes and edges
    nodes = sub_graph[2]
    edges = sub_graph[3]
    edge_type = sub_graph[4]

    # remove (user,bundle) or (bundle,user) from edges to avoid leakage
    if (user_global_id, bundle_global_id) in edges:
        idx = edges.index([user_global_id, bundle_global_id])
        del edges[idx]
        del edge_type[idx]
    if (bundle_global_id, user_global_id) in edges:
        idx = edges.index([bundle_global_id, user_global_id])
        del edges[idx]
        del edge_type[idx]

    if edges == []:
        return
    # reassign the node to 0,1,2....
    node_type = []
    node_dict = defaultdict(int)
    idx = 0
    for node in nodes:
        node_dict[node] = idx
        node_type.append(get_node_type(node, [user_global_id, bundle_global_id]))
        idx += 1

    # reassign the edges
    up = []
    down = []
    # ensure graph is undirected
    new_edges = edges[:]
    for edge in edges:
        inverse_edge = [edge[1], edge[0]]
        if inverse_edge not in edges:
            new_edges.append(inverse_edge)
            edge_type.append(get_edge_type(inverse_edge))
    edges = new_edges
    for edge in edges:
        up.append(node_dict[edge[0]])
        down.append(node_dict[edge[1]])

    # define Data parameters
    x = torch.FloatTensor(get_x(node_type, x_length))

    y = 0
    if bundle_private_id in user_bundle[user_private_id]:
        y = 1
    y = torch.FloatTensor([y])

    # print(up)
    # print(x)
    up = torch.Tensor(up).long()
    down = torch.Tensor(down).long()
    edge_index = torch.stack([up, down], 0)
    edge_type = torch.Tensor(edge_type)

    data = Data(x, edge_index, edge_type=edge_type, y=y)

    return data























