import pickle
import os
import time
from glob import glob
from copy import deepcopy
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import freeze_support
from scipy import sparse, io, stats


data_root = r"E:\SGCN"
num_threads = mp.cpu_count()

print("Threads num: ", num_threads)

# ------------------------------------- Raw Data preparation --------------------------------------------

# Python2 pickle -> Python3 pickle
# Exp: pickle.load(StrToBytes(data_file))


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, size):
        return self.fileobj.read(size).encode()

    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

# USER-ITEM USER-BUNDLE BUNDLE-ITEM mat
user_item = load_obj(os.path.join(
    data_root, r"baseline_data\Youshu\user_item_dict"))
user_bundle = load_obj(os.path.join(
    data_root, r"baseline_data\Youshu\user_bundle_dict"))
bundle_item = load_obj(os.path.join(
    data_root, r"baseline_data\Youshu\bundle_item_dict"))

# BUNDLE-USER mat
bundle_user = defaultdict(list)
for user in user_bundle:
    items = user_bundle[user]
    for item in items:
        if user not in bundle_user[item]:
            bundle_user[item].append(user)

# ITEM-BUNDLE mat
item_bundle = defaultdict(list)
for bundle in bundle_item:
    items = bundle_item[bundle]
    for item in items:
        if bundle not in item_bundle[item]:
            item_bundle[item].append(bundle)

# now we need to unify all the items,bundles,users with coherent IDs
all_user_item = load_obj(
    os.path.join(
        data_root, r"baseline_data\Youshu\user_item"))
all_bundle_item = load_obj(
    os.path.join(
        data_root, r"baseline_data\Youshu\list_item"))

num_items = all_user_item.shape[1]
num_users = all_user_item.shape[0]
num_bundles = all_bundle_item.shape[0]

print("User num : ",num_users)
print("Item num : ",num_items)
print("Bundle num : ",num_bundles)

ids = 0
id_dict = defaultdict(str)
find_id = defaultdict(int)

# users are the first
for user in range(num_users):
    id_dict[ids] = "user"+str(user)
    ids += 1
# bundles are the second
for bundle in range(num_bundles):
    id_dict[ids] = "bundle"+str(bundle)
    ids += 1
# items are the last
for item in range(num_items):
    id_dict[ids] = "item"+str(item)
    ids += 1

for id in id_dict:
    value = id_dict[id]
    find_id[value] = id


# ---------------------------------------- helper functions -------------------------------------------

def AttNum_to_id(attr, num):
    if attr+str(num) not in find_id:
        return None
    return find_id[attr+str(num)]


def id_to_AttNum(id):
    stri = id_dict[id]
    if "item" in stri:
        return ["item", int(stri[4:])]
    elif "bundle" in stri:
        return ["bundle", int(stri[6:])]
    elif "user" in stri:
        return ["user", int(stri[4:])]
    return None


def is_user(id):
    l = id_to_AttNum(id)
    if l == None:
        # print("In is_user, Id not found!")
        return None
    attr, num = l[0], l[1]
    return attr == "user"


def is_bundle(id):
    l = id_to_AttNum(id)
    if l == None:
        # print("In is_bundle, Id not found!")
        return None
    attr, num = l[0], l[1]
    return attr == "bundle"


def is_item(id):
    l = id_to_AttNum(id)
    if l == None:
        # print("In is_item, Id not found!")
        return None
    attr, num = l[0], l[1]
    return attr == "item"


def bundle_neighbors(bundle_id):
    # bundle_id is the private id
    items = bundle_item[bundle_id]
    users = bundle_user[bundle_id]
    item_ids = []
    user_ids = []
    for item in items:
        if AttNum_to_id("item", item) == None:
            continue
        item_ids.append(AttNum_to_id("item", item))
    for user in users:
        if AttNum_to_id("user", user) == None:
            continue
        user_ids.append(AttNum_to_id("user", user))
    return [item_ids, user_ids]


def user_neighbors(user_id):
    # user_id is the private id
    items = user_item[user_id]
    bundles = user_bundle[user_id]
    item_ids = []
    bundle_ids = []
    for item in items:
        if AttNum_to_id("item", item) == None:
            continue
        item_ids.append(AttNum_to_id("item", item))
    for bundle in bundles:
        if AttNum_to_id("bundle", bundle) == None:
            continue
        bundle_ids.append(AttNum_to_id("bundle", bundle))
    return [item_ids, bundle_ids]


def get_edge_type(edge):
    # edge is a list of global ids
    # return None if there is no such kind of edge!!
    node1 = edge[0]
    node2 = edge[1]
    attr1, _ = id_to_AttNum(node1)
    attr2, _ = id_to_AttNum(node2)

    if [attr1, attr2] in [["bundle", "item"], ["item", "bundle"]]:
        return 0
    elif [attr1, attr2] in [["bundle", "user"], ["user", "bundle"]]:
        return 1
    elif [attr1, attr2] in [["user", "item"], ["item", "user"]]:
        return 2

    return None


def get_node_type(node, pair):
    # node is the global id of a node
    # pair is the current (user,bundle) pair of a subgraph
    attr, _ = id_to_AttNum(node)
    if attr == "bundle" and node in pair:
        return 0
    elif attr == "user" and node in pair:
        return 1
    elif attr == "bundle":
        return 2
    elif attr == "user":
        return 3
    elif attr == "item":
        return 4


# ---------------------- now we construct k-hop subgraphs centralled with a (user,bundle) pair ---------------------

# helper function to find k_hop neighbors centered at a (user,bundle) global id pair
def k_hops(k, user_id, bundle_id):
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
                        if item not in nodes:
                            nodes.append(item)
                        if [node, item] not in edges:
                            edges.append([node, item])
                            edge_type.append(get_edge_type([node,item]))

                # user will be added to cur_layer
                if user_neighbor != []:
                    for user in user_neighbor:
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
                        if item not in nodes:
                            nodes.append(item)
                        if [node, item] not in edges:
                            edges.append([node, item])
                            edge_type.append(get_edge_type([node, item]))

                # user will be added to cur_layer
                if bundle_neighbor != []:
                    for bundle in bundle_neighbor:
                        if bundle not in nodes:
                            nodes.append(bundle)
                            cur_layer.append(bundle)
                        if [node, bundle] not in edges:
                            edges.append([node, bundle])
                            edge_type.append(get_edge_type([node, bundle]))

        last_layer = cur_layer
        cur_layer = []

    
    return nodes,edges,edge_type


# to extract all subgraphs


def extract_subgraphs_thread(hop=1, user_range=range(num_users), thread_id=1):
    # the format will be {(user,bundle):subgraph....}
    subgraph_dict = []
    # print(f"This is thread {thread_id}")
    for user_pid in tqdm(user_range):
        for bundle_pid in range(num_bundles):
            subgraph = k_hops(hop, AttNum_to_id("user", user_pid),
                              AttNum_to_id("bundle", bundle_pid))
            if subgraph[1] != []:
                subgraph_dict.append(
                    [AttNum_to_id("user", user_pid), AttNum_to_id("bundle", bundle_pid), subgraph[0], subgraph[1],subgraph[2]])

    return subgraph_dict





