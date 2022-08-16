import pickle
import sys
import os
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset, InMemoryDataset
# from train import *
sys.path.extend([r"E:\SGCN\utils"])
from baseline_preprocess_Youshu import *
from render_dataset import *

def need_drop(prop):
    m = np.random.random()
    if m >= 0 and m <= prop:
        return 1
    else:
        return 0

def k_hops_for_test(k, user_id, bundle_id,drop_prop):
    nodes = [user_id, bundle_id]
    edges = []
    edge_type = []
    last_layer = [user_id, bundle_id]
    cur_layer = []
    for hop in range(k):
        for node in last_layer:
            if is_bundle(node):
                _, bundle_id = id_to_AttNum(node)
                l = bundle_neighbors(bundle_id)
                item_neighbor = l[0]
                user_neighbor = l[1]

                # item will not be added to cur_layer
                if item_neighbor != []:
                    for item in item_neighbor:
                        if need_drop(drop_prop):
                            continue
                        if item not in nodes:
                            nodes.append(item)
                        if [node, item] not in edges:
                            edges.append([node, item])
                            edge_type.append(get_edge_type([node, item]))

                # user will be added to cur_layer
                if user_neighbor != []:
                    for user in user_neighbor:
                        if need_drop(drop_prop):
                            continue

                        if user not in nodes:
                            nodes.append(user)
                            cur_layer.append(user)
                        if [node, user] not in edges:
                            edges.append([node, user])
                            edge_type.append(get_edge_type([node, user]))

            elif is_user(node):
                _, user_id = id_to_AttNum(node)
                l = user_neighbors(user_id)
                item_neighbor = l[0]
                bundle_neighbor = l[1]

                # item will not be added to cur_layer
                if item_neighbor != []:
                    for item in item_neighbor:
                        if need_drop(drop_prop):
                            continue
                        if item not in nodes:
                            nodes.append(item)
                        if [node, item] not in edges:
                            edges.append([node, item])
                            edge_type.append(get_edge_type([node, item]))

                # user will be added to cur_layer
                if bundle_neighbor != []:
                    for bundle in bundle_neighbor:
                        if need_drop(drop_prop):
                            continue
                        if bundle not in nodes:
                            nodes.append(bundle)
                            cur_layer.append(bundle)
                        if [node, bundle] not in edges:
                            edges.append([node, bundle])
                            edge_type.append(get_edge_type([node, bundle]))

        last_layer = cur_layer
        cur_layer = []

    return nodes, edges, edge_type



def extract_subgraphs_for_test(hop=1, user_range=range(num_users), drop_prop=0.5):
    # the format will be {(user,bundle):subgraph....}
    subgraph_dict = []
    # print(f"This is thread {thread_id}")
    for user_pid in tqdm(user_range):
        for bundle_pid in range(num_bundles):
            subgraph = k_hops_for_test(hop, AttNum_to_id("user", user_pid),
                              AttNum_to_id("bundle", bundle_pid),drop_prop)
            if subgraph[1] != []:
                subgraph_dict.append(
                    [AttNum_to_id("user", user_pid), AttNum_to_id("bundle", bundle_pid), subgraph[0], subgraph[1],subgraph[2]])

    return subgraph_dict



# test_users = np.load(r"E:\SGCN\baseline_data\Youshu\train_test_data\4\r_test_users.npy")
# subgraphs = extract_subgraphs_for_test(1,test_users,0.25)

# test for average rmse

# 0.25

