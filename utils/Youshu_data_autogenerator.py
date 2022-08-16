import pickle
import sys
import os
import numpy as np
from baseline_preprocess_Youshu import *
from render_dataset import *
from negative_sample import *
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset, InMemoryDataset
sys.path.extend(
    [data_root])
from model import *

train_ratio = 0.8

file_num = 4
os.chdir(os.path.join(data_root, r"baseline_data\Youshu\train_test_data"))
if not os.path.exists(str(file_num)):
    os.mkdir(str(file_num))

# this part should only be ignited once

# all_users = np.arange(num_users)
# np.random.shuffle(all_users)
# np.save(str(file_num) + r"/r_all_users.npy", all_users)
# train_users = all_users[:int(train_ratio*num_users)]
# test_users = all_users[int(train_ratio*num_users):]
# np.save(str(file_num)+ r"/r_train_users.npy", train_users)
# np.save(str(file_num)+ r"/r_test_users.npy", test_users)
#
# num_train = len(train_users)
# num_test = len(test_users)


dict1 = extract_subgraphs_thread(1,user_range=train_users[:int(0.2*num_train)])
save_obj(dict1, str(file_num) + r"/train_data_1_stage1.pkl")
del dict1

dict1 = extract_subgraphs_thread(1,user_range=train_users[int(0.2*num_train):int(0.4*num_train)])
save_obj(dict1, str(file_num) + r"/train_data_2_stage1.pkl")
del dict1

dict1 = extract_subgraphs_thread(1,user_range=train_users[int(0.4*num_train):int(0.6*num_train)])
save_obj(dict1, str(file_num) + r"/train_data_3_stage1.pkl")
del dict1

dict1 = extract_subgraphs_thread(1,user_range=train_users[int(0.6*num_train):int(0.8*num_train)])
save_obj(dict1, str(file_num) + r"/train_data_4_stage1.pkl")
del dict1

dict1 = extract_subgraphs_thread(1,user_range=train_users[int(0.8*num_train):])
save_obj(dict1, str(file_num) + r"/train_data_5_stage1.pkl")
del dict1

dict2 = extract_subgraphs_thread(1,user_range=test_users)
save_obj(dict2, str(file_num) + r"/test_data_stage1.pkl")
del dict2

print()
print("Stage1 Finished")
print("-----------------------------------------------------------------------")





print()
print("Stage2: render torch_geometric.data")
print("-----------------------------------------------------------------------")

# -------------------------------------------------------------------------------------------------------------
print()
print("train_data_1_stage2: start")
f1 = os.chdir(os.path.join(data_root, r"baseline_data\Youshu\train_test_data"))
f2 = str(file_num)
f3 = "train_data_1_stage1.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "rb")
raw = pickle.load(f)
f.close()

max = len(raw)
start = 0
end = 1000000
while True:
    datas = []
    if end >= max and start < max:
        end = max
        for subgraph in tqdm(raw[start:end]):
            datas.append(construct_graph_data(subgraph))
        f3 = "train_data_1_stage2_" + str(start) + "_" + str(end) + ".pkl"
        f = open(
            os.path.join(f1,f2,f3),
            "wb")
        pickle.dump(datas, f)
        f.close()

        del datas
        break
    else:
        for subgraph in tqdm(raw[start:end]):
            datas.append(construct_graph_data(subgraph))
        f3 = "train_data_1_stage2_" + str(start) + "_" + str(end) + ".pkl"
        f = open(
            os.path.join(f1, f2, f3),
            "wb")
        pickle.dump(datas, f)
        f.close()

        del datas
        start = end
        end = end + 1000000
        continue
del raw
print("train_data_1_stage2: finished")

# -------------------------------------------------------------------------------------------------------------
print()
print("train_data_2_stage2: start")
f1 = os.chdir(os.path.join(data_root, r"baseline_data\Youshu\train_test_data"))
f2 = str(file_num)
f3 = "train_data_2_stage1.pkl"
f4 = os.path.join(f1, f2, f3)
f = open(f4, "rb")
raw = pickle.load(f)
f.close()

max = len(raw)
start = 0
end = 1000000
while True:
    datas = []
    if end >= max and start < max:
        end = max
        for subgraph in tqdm(raw[start:end]):
            datas.append(construct_graph_data(subgraph))
        f3 = "train_data_2_stage2_" + str(start) + "_" + str(end) + ".pkl"
        f = open(
            os.path.join(f1, f2, f3),
            "wb")
        pickle.dump(datas, f)
        f.close()

        del datas
        break
    else:
        for subgraph in tqdm(raw[start:end]):
            datas.append(construct_graph_data(subgraph))
        f3 = "train_data_2_stage2_" + str(start) + "_" + str(end) + ".pkl"
        f = open(
            os.path.join(f1, f2, f3),
            "wb")
        pickle.dump(datas, f)
        f.close()

        del datas
        start = end
        end = end + 1000000
        continue

del raw
print("train_data_2_stage2: finished")

# -------------------------------------------------------------------------------------------------------------
print()
print("train_data_3_stage2: start")
f1 = os.chdir(os.path.join(data_root, r"baseline_data\Youshu\train_test_data"))
f2 = str(file_num)
f3 = "train_data_3_stage1.pkl"
f4 = os.path.join(f1, f2, f3)
f = open(f4, "rb")
raw = pickle.load(f)
f.close()

max = len(raw)
start = 0
end = 1000000
while True:
    datas = []
    if end >= max and start < max:
        end = max
        for subgraph in tqdm(raw[start:end]):
            datas.append(construct_graph_data(subgraph))
        f3 = "train_data_3_stage2_" + str(start) + "_" + str(end) + ".pkl"
        f = open(
            os.path.join(f1, f2, f3),
            "wb")
        pickle.dump(datas, f)
        f.close()

        del datas
        break
    else:
        for subgraph in tqdm(raw[start:end]):
            datas.append(construct_graph_data(subgraph))
        f3 = "train_data_3_stage2_" + str(start) + "_" + str(end) + ".pkl"
        f = open(
            os.path.join(f1, f2, f3),
            "wb")
        pickle.dump(datas, f)
        f.close()

        del datas
        start = end
        end = end + 1000000
        continue

del raw
print("train_data_3_stage2: finished")

# -------------------------------------------------------------------------------------------------------------
print()
print("train_data_4_stage2: start")
f1 = os.chdir(os.path.join(data_root, r"baseline_data\Youshu\train_test_data"))
f2 = str(file_num)
f3 = "train_data_4_stage1.pkl"
f4 = os.path.join(f1, f2, f3)
f = open(f4, "rb")
raw = pickle.load(f)
f.close()

max = len(raw)
start = 0
end = 1000000
while True:
    datas = []
    if end >= max and start < max:
        end = max
        for subgraph in tqdm(raw[start:end]):
            datas.append(construct_graph_data(subgraph))
        f3 = "train_data_4_stage2_" + str(start) + "_" + str(end) + ".pkl"
        f = open(
            os.path.join(f1, f2, f3),
            "wb")
        pickle.dump(datas, f)
        f.close()

        del datas
        break
    else:
        for subgraph in tqdm(raw[start:end]):
            datas.append(construct_graph_data(subgraph))
        f3 = "train_data_4_stage2_" + str(start) + "_" + str(end) + ".pkl"
        f = open(
            os.path.join(f1, f2, f3),
            "wb")
        pickle.dump(datas, f)
        f.close()

        del datas
        start = end
        end = end + 1000000
        continue

del raw
print("train_data_4_stage2: finished")

# -------------------------------------------------------------------------------------------------------------
print()
print("train_data_5_stage2: start")
f1 = os.chdir(os.path.join(data_root, r"baseline_data\Youshu\train_test_data"))
f2 = str(file_num)
f3 = "train_data_5_stage1.pkl"
f4 = os.path.join(f1, f2, f3)
f = open(f4, "rb")
raw = pickle.load(f)
f.close()

max = len(raw)
start = 0
end = 1000000
while True:
    datas = []
    if end >= max and start < max:
        end = max
        for subgraph in tqdm(raw[start:end]):
            datas.append(construct_graph_data(subgraph))
        f3 = "train_data_5_stage2_" + str(start) + "_" + str(end) + ".pkl"
        f = open(
            os.path.join(f1, f2, f3),
            "wb")
        pickle.dump(datas, f)
        f.close()

        del datas
        break
    else:
        for subgraph in tqdm(raw[start:end]):
            datas.append(construct_graph_data(subgraph))
        f3 = "train_data_5_stage2_" + str(start) + "_" + str(end) + ".pkl"
        f = open(
            os.path.join(f1, f2, f3),
            "wb")
        pickle.dump(datas, f)
        f.close()

        del datas
        start = end
        end = end + 1000000
        continue

del raw
print("train_data_5_stage2: finished")

# -------------------------------------------------------------------------------------------------------------
print()
print("test_data_stage2: start")
f1 = os.chdir(os.path.join(data_root, r"baseline_data\Youshu\train_test_data"))
f2 = str(file_num)
f3 = "test_data_stage1.pkl"
f4 = os.path.join(f1, f2, f3)
f = open(f4, "rb")
raw = pickle.load(f)
f.close()

max = len(raw)
start = 0
end = 1000000
while True:
    datas = []
    if end >= max and start < max:
        end = max
        for subgraph in tqdm(raw[start:end]):
            datas.append(construct_graph_data(subgraph))
        f3 = "test_data_stage2_" + str(start) + "_" + str(end) + ".pkl"
        f = open(
            os.path.join(f1, f2, f3),
            "wb")
        pickle.dump(datas, f)
        f.close()

        del datas
        break
    else:
        for subgraph in tqdm(raw[start:end]):
            datas.append(construct_graph_data(subgraph))
        f3 = "test_data_stage2_" + str(start) + "_" + str(end) + ".pkl"
        f = open(
            os.path.join(f1, f2, f3),
            "wb")
        pickle.dump(datas, f)
        f.close()

        del datas
        start = end
        end = end + 1000000
        continue

del raw
print("test_data_stage2: finished")

print()
print("Stage2 Finished")
print("-----------------------------------------------------------------------")
