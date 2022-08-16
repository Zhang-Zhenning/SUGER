import pickle
import sys
import os
import numpy as np
# data_root = r"C:\Users\BOURNE\Desktop\zzn_project\SGCN"
# sys.path.extend([os.path.join(data_root, r"utils")])
from baseline_preprocess_Youshu import *
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset, InMemoryDataset

def locate_pos(subgraphs):
    # return a list of 0,1 sequence
    # indicating whether the subgraph is pos or neg
    positive = []
    user_bundle_pairs = []
    
    # loop all the subgraphs
    for subgraph in subgraphs:
        user_global_id = subgraph[0]
        bundle_global_id = subgraph[1]

        if is_bundle(user_global_id):
            user_global_id = subgraph[1]
            bundle_global_id = subgraph[0]
        
        user_bundle_pairs.append([user_global_id,bundle_global_id])
        
        _, user_private_id = id_to_AttNum(user_global_id)
        _, bundle_private_id = id_to_AttNum(bundle_global_id)

        if bundle_private_id in user_bundle[user_private_id]:
            positive.append(1)
        
        else:
            positive.append(0)
    
    return positive,user_bundle_pairs

def generate_selection(subgraphs,n_p_ratio=2):
    # return a list of 0,1 sequence to indicate 
    # which subgraphs we need to use in the training phase

    positive,user_bundle_pairs = locate_pos(subgraphs)
    positive = np.array(positive)

    num_pos = np.sum(positive)
    if n_p_ratio == "all":
        num_neg = len(positive) - num_pos - 1
    else:
        num_neg = int(num_pos * n_p_ratio)

    total_neg = len(positive) - num_pos

    select_map = np.zeros(len(positive))
    
    # select all positive samples
    select_map[positive==1] = 2

    # randomly select negatives
    neg_map = np.zeros(total_neg)
    neg_map[:num_neg] = 1
    np.random.shuffle(neg_map)

    i = 0
    j = 0

    for i in range(len(positive)):
        if select_map[i] == 0:
            if neg_map[j] == 1:
                select_map[i] = 1
                j += 1
            else:
                j += 1
        else:
            continue
    
    return select_map.astype(np.int32),user_bundle_pairs

def basic_neg_sample(subgraphs,datas,n_p_ratio=2):
    # subgraphs: a list of the raw data in processed data, description is in README
    # datas: a list of torch_geometric.data.Data items corresponding to subgraphs to be selected
    select_map,user_bundle_pairs = generate_selection(subgraphs,n_p_ratio)
    final_datas = []
    final_ub_pairs = []
    labels = []

    for i in range(select_map.shape[0]):
        if select_map[i] == 1:
            final_datas.append(datas[i])
            final_ub_pairs.append(user_bundle_pairs[i])
            labels.append(0)
        elif select_map[i] == 2:
            final_datas.append(datas[i])
            final_ub_pairs.append(user_bundle_pairs[i])
            labels.append(1)
     
    return final_datas,final_ub_pairs,labels












 










        



