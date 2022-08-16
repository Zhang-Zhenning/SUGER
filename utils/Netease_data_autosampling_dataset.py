import pickle
import sys
import os
import numpy as np
from baseline_preprocess_Netease import *
from render_dataset_Netease import *
from negative_sample_Netease import *
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset, InMemoryDataset
sys.path.extend(
    [data_root])
from model import *


file_num = 2

print()
print("Stage3: Negative sampling")
print("-----------------------------------------------------------------------")

# -------------------------------------------------------------------------------------------------------------
print()
print("train_data_1_stage3: start")
f1 = os.path.join(data_root, r"baseline_data\Netease\train_test_data")
f2 = str(file_num)
f3 = "train_data_1_stage1.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "rb")
stage1_data = pickle.load(f)
f.close()

f1 = os.path.join(data_root, r"baseline_data\Netease\train_test_data")
f2 = str(file_num)
f3 = "train_data_1_stage2*.pkl"
f4 = os.path.join(f1,f2,f3)

# TODO: need check the order of sorted
stage2_data = []
for file in sorted(glob(f4)):
    f = open(file, "rb")
    temp_data = pickle.load(f)
    f.close()
    stage2_data = stage2_data + temp_data
    del temp_data

if len(stage1_data) != len(stage2_data):
    print("WARNING: two stage datas are inconsistent!")

data_s, pair_s, label_s = basic_neg_sample(stage1_data, stage2_data, 2)

del stage1_data
del stage2_data
#
f1 = os.path.join(data_root, r"baseline_data\Netease\train_test_data")
f2 = str(file_num)
f3 = "train_data_1_stage3_datas.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "wb")
pickle.dump(data_s, f)
f.close()

del data_s

f3 = "train_data_1_stage3_pairs.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "wb")
pickle.dump(pair_s, f)
f.close()

del pair_s

f3 = "train_data_1_stage3_labels.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "wb")
pickle.dump(label_s, f)
f.close()

del label_s
print("train_data_1_stage3: finished")
#
# # -------------------------------------------------------------------------------------------------------------
print()
print("train_data_2_stage3: start")
f1 = os.path.join(data_root, r"baseline_data\Netease\train_test_data")
f2 = str(file_num)
f3 = "train_data_2_stage1.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "rb")
stage1_data = pickle.load(f)
f.close()

f1 = os.path.join(data_root, r"baseline_data\Netease\train_test_data")
f2 = str(file_num)
f3 = "train_data_2_stage2*.pkl"
f4 = os.path.join(f1,f2,f3)

# TODO: need check the order of sorted
stage2_data = []
for file in sorted(glob(f4)):
    f = open(file, "rb")
    temp_data = pickle.load(f)
    f.close()

    stage2_data = stage2_data + temp_data
    del temp_data

if len(stage1_data) != len(stage2_data):
    print("WARNING: two stage datas are inconsistent!")

data_s, pair_s, label_s = basic_neg_sample(stage1_data, stage2_data, 2)

del stage1_data
del stage2_data

f1 = os.path.join(data_root, r"baseline_data\Netease\train_test_data")
f2 = str(file_num)
f3 = "train_data_2_stage3_datas.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "wb")
pickle.dump(data_s, f)
f.close()

del data_s

f3 = "train_data_2_stage3_pairs.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "wb")
pickle.dump(pair_s, f)
f.close()

del pair_s

f3 = "train_data_2_stage3_labels.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "wb")
pickle.dump(label_s, f)
f.close()

del label_s
print("train_data_2_stage3: finished")

# -------------------------------------------------------------------------------------------------------------
print()
print("train_data_3_stage3: start")
f1 = os.path.join(data_root, r"baseline_data\Netease\train_test_data")
f2 = str(file_num)
f3 = "train_data_3_stage1.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "rb")
stage1_data = pickle.load(f)
f.close()

f1 = os.path.join(data_root, r"baseline_data\Netease\train_test_data")
f2 = str(file_num)
f3 = "train_data_3_stage2*.pkl"
f4 = os.path.join(f1,f2,f3)

# TODO: need check the order of sorted
stage2_data = []
for file in sorted(glob(f4)):
    f = open(file, "rb")
    temp_data = pickle.load(f)
    f.close()

    stage2_data = stage2_data + temp_data
    del temp_data

if len(stage1_data) != len(stage2_data):
    print("WARNING: two stage datas are inconsistent!")

data_s, pair_s, label_s = basic_neg_sample(stage1_data, stage2_data, 2)

del stage1_data
del stage2_data

f1 = os.path.join(data_root, r"baseline_data\Netease\train_test_data")
f2 = str(file_num)
f3 = "train_data_3_stage3_datas.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "wb")
pickle.dump(data_s, f)
f.close()

del data_s

f3 = "train_data_3_stage3_pairs.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "wb")
pickle.dump(pair_s, f)
f.close()

del pair_s

f3 = "train_data_3_stage3_labels.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "wb")
pickle.dump(label_s, f)
f.close()

del label_s
print("train_data_3_stage3: finished")

# -------------------------------------------------------------------------------------------------------------
print()
print("train_data_4_stage3: start")
f1 = os.path.join(data_root, r"baseline_data\Netease\train_test_data")
f2 = str(file_num)
f3 = "train_data_4_stage1.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "rb")
stage1_data = pickle.load(f)
f.close()

f1 = os.path.join(data_root, r"baseline_data\Netease\train_test_data")
f2 = str(file_num)
f3 = "train_data_4_stage2*.pkl"
f4 = os.path.join(f1,f2,f3)

# TODO: need check the order of sorted
stage2_data = []
for file in sorted(glob(f4)):
    f = open(file, "rb")
    temp_data = pickle.load(f)
    f.close()

    stage2_data = stage2_data + temp_data
    del temp_data

if len(stage1_data) != len(stage2_data):
    print("WARNING: two stage datas are inconsistent!")

data_s, pair_s, label_s = basic_neg_sample(stage1_data, stage2_data, 2)

del stage1_data
del stage2_data

f1 = os.path.join(data_root, r"baseline_data\Netease\train_test_data")
f2 = str(file_num)
f3 = "train_data_4_stage3_datas.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "wb")
pickle.dump(data_s, f)
f.close()

del data_s

f3 = "train_data_4_stage3_pairs.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "wb")
pickle.dump(pair_s, f)
f.close()

del pair_s

f3 = "train_data_4_stage3_labels.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "wb")
pickle.dump(label_s, f)
f.close()

del label_s
print("train_data_4_stage3: finished")

# -------------------------------------------------------------------------------------------------------------
print()
print("train_data_5_stage3: start")
f1 = os.path.join(data_root, r"baseline_data\Netease\train_test_data")
f2 = str(file_num)
f3 = "train_data_5_stage1.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "rb")
stage1_data = pickle.load(f)
f.close()

f1 = os.path.join(data_root, r"baseline_data\Netease\train_test_data")
f2 = str(file_num)
f3 = "train_data_5_stage2*.pkl"
f4 = os.path.join(f1,f2,f3)

# TODO: need check the order of sorted
stage2_data = []
for file in sorted(glob(f4)):
    f = open(file, "rb")
    temp_data = pickle.load(f)
    f.close()

    stage2_data = stage2_data + temp_data
    del temp_data

if len(stage1_data) != len(stage2_data):
    print("WARNING: two stage datas are inconsistent!")

data_s, pair_s, label_s = basic_neg_sample(stage1_data, stage2_data, 2)

del stage1_data
del stage2_data

f1 = os.path.join(data_root, r"baseline_data\Netease\train_test_data")
f2 = str(file_num)
f3 = "train_data_5_stage3_datas.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "wb")
pickle.dump(data_s, f)
f.close()

del data_s

f3 = "train_data_5_stage3_pairs.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "wb")
pickle.dump(pair_s, f)
f.close()

del pair_s

f3 = "train_data_5_stage3_labels.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "wb")
pickle.dump(label_s, f)
f.close()

del label_s
print("train_data_5_stage3: finished")

# -------------------------------------------------------------------------------------------------------------
print()
print("test_data_stage3: start")
f1 = os.path.join(data_root, r"baseline_data\Netease\train_test_data")
f2 = str(file_num)
f3 = "test_data_stage1.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "rb")
stage1_data = pickle.load(f)
f.close()

f1 = os.path.join(data_root, r"baseline_data\Netease\train_test_data")
f2 = str(file_num)
f3 = "test_data_stage2*.pkl"
f4 = os.path.join(f1,f2,f3)

# TODO: need check the order of sorted
stage2_data = []
for file in sorted(glob(f4)):
    f = open(file, "rb")
    temp_data = pickle.load(f)
    f.close()

    stage2_data = stage2_data + temp_data
    del temp_data

if len(stage1_data) != len(stage2_data):
    print("WARNING: two stage datas are inconsistent!")

data_s, pair_s, label_s = basic_neg_sample(stage1_data, stage2_data, 2)

del stage1_data
del stage2_data

f1 = os.path.join(data_root, r"baseline_data\Netease\train_test_data")
f2 = str(file_num)
f3 = "test_data_stage3_datas.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "wb")
pickle.dump(data_s, f)
f.close()

del data_s

f3 = "test_data_stage3_pairs.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "wb")
pickle.dump(pair_s, f)
f.close()

del pair_s

f3 = "test_data_stage3_labels.pkl"
f4 = os.path.join(f1,f2,f3)
f = open(f4, "wb")
pickle.dump(label_s, f)
f.close()

del label_s
print("test_data_stage3: finished")
