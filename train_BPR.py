from model import BasicModel,EMB_Model
import os
import time
import numpy as np
import math
import multiprocessing as mp
import networkx as nx
import torch
import torch.nn.functional as F
import pickle
import random

from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data, Dataset, DataLoader, DenseDataLoader, InMemoryDataset
from tqdm import tqdm
from glob import glob

# select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def BPRloss_emb(r1, r2, model, lamb,device):
    loss = torch.tensor([0]).to(device).float()
    for i in model.parameters():
         loss += torch.norm(i)
    
    loss = -torch.log(torch.sigmoid(torch.norm(r1-r2))) + lamb * loss
    return loss


def BPRloss(r1, r2, model, lamb, device):
    loss = torch.tensor([0]).to(device).float()
    for i in model.parameters():
        loss += torch.norm(i)

    loss = -torch.log(torch.sigmoid(r1-r2)) + lamb * loss
    return loss


def BPRloss_penalty(r1, r2, model, lamb, device):
    loss = torch.tensor([0]).to(device).float()
    for i in model.parameters():
        loss += torch.norm(i)

    loss = -torch.log(torch.sigmoid(r1-r2)) + lamb * loss + torch.norm(1/(r1 - r2))
    return loss

# compute loss and rmse
def eval_loss_emb(model, loader, device, regression=False, show_progress=False):
    model.eval()
    loss = 0
    bad = 0
    if show_progress:
        print('Testing begins...')
        pbar = tqdm(loader)
    else:
        pbar = loader
    for data in pbar:
        data = data.to(device)
        with torch.no_grad():
            out,_ = model(data)
        if out == None:
            bad += 1
            continue
        if regression:
            # print(out,data.y.view(-1))
            l =  F.mse_loss(out, data.y.view(-1), reduction='sum').item()
            loss += l
            print(out, data.y.view(-1),l,end="")
        else:
            loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
        torch.cuda.empty_cache()
    print(len(loader.dataset))
    return loss / (len(loader.dataset) - bad), bad


#  compute loss and rmse
def eval_loss(model, loader, device, regression=False, show_progress=False):
    # model.eval()
    loss = 0
    bad = 0
    if show_progress:
        # print('Testing begins...')
        pbar = tqdm(loader)
    else:
        pbar = loader
    for data in pbar:
        data = data.to(device)

        out= model(data)
        if out == None:
            bad += 1
            continue
        if regression:
            # print(out,data.y.view(-1))
            l = F.mse_loss(out, data.y.view(-1), reduction='sum').item()
            loss += l
            # if data.y.view(-1)[0] == 1:
            #     continue
            print(out, data.y.view(-1), l)
        else:
            loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
        torch.cuda.empty_cache()
    print(len(loader.dataset))
    return loss / (len(loader.dataset) - bad), bad


def get_rmse(model, loader, device, show_progress=False):
    mse_loss, bad = eval_loss(model, loader, device, True, show_progress)
    rmse = math.sqrt(mse_loss)
    return rmse, bad


def get_rmse_emb(model, loader, device, show_progress=False):
    mse_loss, bad = eval_loss_emb(model, loader, device, True, show_progress)
    rmse = math.sqrt(mse_loss)
    return rmse, bad

# define train once process


def train(model,
          optimizer,
          dataloader_p,
          dataloader_n,
          device,
          regression=True,
          ARR=0,
          show_progress=True,
          epoch=None):

    # open batch normalization
    model.train()
    total_loss = 0

    # show progress or not

    bad = 0
    i = 0
    # print("---Start Training---")

    for databatch1,databatch2 in zip(dataloader_p,dataloader_n):
        i += 1
        # pbar.update(1)
        # databatch.to(device)
        optimizer.zero_grad()
        databatch1 = databatch1.to(device)
        databatch2 = databatch2.to(device)
        result1 = model(databatch1)
        result2 = model(databatch2)

        if result1 is None or result2 is None:
            bad += 1
            continue


        cur_loss = BPRloss(result1,result2,model,0.00001,device)
        cur_loss.backward()

        if databatch1.batch is not None and databatch2.batch is not None:
            num_graph = databatch1.num_graphs + databatch2.num_graphs
        else:
            num_graph = databatch1.x.size(0) + databatch2.x.size(0)

        total_loss += cur_loss.item()

        optimizer.step()
        torch.cuda.empty_cache()

            
    # print("The bad training batch is: ", bad)

    return total_loss / (len(dataloader_p.dataset) + len(dataloader_n.dataset))


# define bpr train
def trainEMB(model,
          optimizer,
          dataloader_p,
          dataloader_n,
          device,
          regression=True,
          ARR=0,
          show_progress=True,
          epoch=None):

    # open batch normalization
    model.train()
    total_loss = 0

    # show progress or not

    bad = 0
    i = 0
    # print("---Start Training---")

    with tqdm(total=len(dataloader_p.dataset)) as pbar:
        for databatch1, databatch2 in zip(dataloader_p, dataloader_n):
            i += 1
            pbar.update(1)
            # databatch.to(device)
            optimizer.zero_grad()
            databatch1 = databatch1.to(device)
            databatch2 = databatch2.to(device)
            result1,emb1 = model(databatch1)
            result2,emb2 = model(databatch2)

            if result1 is None or result2 is None:
                bad += 1
                continue

            cur_loss = BPRloss_emb(emb1, emb2, model, 0.00001, device) * 100000
            cur_loss.backward()

            if databatch1.batch is not None and databatch2.batch is not None:
                num_graph = databatch1.num_graphs + databatch2.num_graphs
            else:
                num_graph = databatch1.x.size(0) + databatch2.x.size(0)

            total_loss += cur_loss.item() * num_graph

            optimizer.step()
            torch.cuda.empty_cache()

    print("The bad training batch is: ", bad)

    return total_loss / (len(dataloader_p.dataset) + len(dataloader_n.dataset))


# define whole train process
def train_process(train_dataset_pos,
                  train_dataset_neg,
                  test_dataset,
                  model,
                  epochs,
                  batch_size,
                  lr,
                  lr_decay_factor,
                  lr_decay_step_size,
                  weight_decay,
                  ARR=0,
                  test_freq=1,
                  show_progress=True):

    # prepare thread and dataloader
    train_loader_p = DataLoader(
        train_dataset_pos, batch_size=batch_size, shuffle=False)
    train_loader_n = DataLoader(
        train_dataset_neg, batch_size=batch_size, shuffle=False)
    # do not need to shuffle test data
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    # prepare model
    model = model.to(device)
    model.reset_parameters()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    cur_epoch = 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if show_progress:
        pbar = tqdm(range(cur_epoch, epochs + cur_epoch))
    else:
        pbar = range(cur_epoch, epochs + cur_epoch)

    rmses = []

    start_time = time.perf_counter()

    for epoch in pbar:
        trainloss = train(model, optimizer, train_loader_p,train_loader_n, device,
                          regression=True, ARR=ARR, show_progress=show_progress, epoch=epoch)
        cur_bad = 0
        # if epoch % test_freq == 0:
        #     cur_rmse, cur_bad = get_rmse(
        #         model, test_loader, device, show_progress=show_progress)
        #     rmses.append(cur_rmse)
        # else:
        #     rmses.append(np.nan)

        if show_progress:
            pbar.set_description(
                'Epoch {}, train loss {:.6f}'.format(
                    epoch, trainloss)
            )

        # if epoch % lr_decay_step_size == 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr_decay_factor * param_group['lr']
    
    torch.save(model, "my_model_4_BPR.pth")

    cur_rmse, cur_bad = get_rmse(
                model, test_loader, device, show_progress=show_progress)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    time_interval = end_time - start_time

    print('Test RMSE:  {:.6f}, Duration: {:.6f}'.format(
        cur_rmse, time_interval))

# define whole train process


def train_process_EMB(train_dataset_pos,
                  train_dataset_neg,
                  test_dataset,
                  model,
                  epochs,
                  batch_size,
                  lr,
                  lr_decay_factor,
                  lr_decay_step_size,
                  weight_decay,
                  ARR=0,
                  test_freq=1,
                  show_progress=True):

    # prepare thread and dataloader
    train_loader_p = DataLoader(
        train_dataset_pos, batch_size=batch_size, shuffle=False)
    train_loader_n = DataLoader(
        train_dataset_neg, batch_size=batch_size, shuffle=False)
    # do not need to shuffle test data
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    # prepare model
    model = model.to(device)
    model.reset_parameters()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    cur_epoch = 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if show_progress:
        pbar = tqdm(range(cur_epoch, epochs + cur_epoch))
    else:
        pbar = range(cur_epoch, epochs + cur_epoch)


    start_time = time.perf_counter()

    for epoch in pbar:
        trainloss = trainEMB(model, optimizer, train_loader_p, train_loader_n, device,
                          regression=True, ARR=ARR, show_progress=show_progress, epoch=epoch)
        cur_bad = 0
        # if epoch % test_freq == 0:
        #     cur_rmse, cur_bad = get_rmse(
        #         model, test_loader, device, show_progress=show_progress)
        #     rmses.append(cur_rmse)
        # else:
        #     rmses.append(np.nan)

        if show_progress:
            pbar.set_description(
                'Epoch {}, train loss {:.6f}'.format(
                    epoch, trainloss)
            )

        # if epoch % lr_decay_step_size == 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr_decay_factor * param_group['lr']
    torch.save(model, "my_model_4_BPR.pth")

    cur_rmse, cur_bad = get_rmse_emb(
        model, test_loader, device, show_progress=show_progress)


    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    time_interval = end_time - start_time

    print('Test RMSE:  {:.6f}, Duration: {:.6f}'.format(
        cur_rmse, time_interval))


def prepare_bpr_dataset_dict(datas,labels,pairs):
 
    pos_pairs = []
    pos_loc = []

    neg_pairs = []
    neg_loc = []

    # collect pos and neg pairs
    for i,data in enumerate(datas):
        if labels[i] == 1:
            pos_loc.append(1)
            neg_loc.append(0)
            pos_pairs.append([pairs[i],datas[i]])
           
        else:
            pos_loc.append(0)
            neg_loc.append(1)
            neg_pairs.append([pairs[i], datas[i]])
    

    # sort
    pos_pairs.sort(key=lambda x:(x[0][0],x[0][1]))
    neg_pairs.sort(key=lambda x:(x[0][0],x[0][1]))

    final_neg_pairs = []
    
    neg_cursor = 0
    for i,pairs in enumerate(pos_pairs):
        cur_pos_user = pairs[0][0]
        while neg_cursor < len(neg_pairs):
            if cur_pos_user == neg_pairs[neg_cursor][0][0]:
                final_neg_pairs.append(neg_pairs[neg_cursor])
                neg_cursor += 1
                break
            else:
                neg_cursor += 1
    
    if len(final_neg_pairs) < len(pos_pairs):
        print("The num of negtive samples is not enough!!!")
    
    return pos_pairs,final_neg_pairs
        




    


# f = open(r"C:\Users\BOURNE\Desktop\zzn_project\Bundle_Recommendation\dataset\current_usage\cur_train_datas.pkl", "rb")
# j = pickle.load(f)
# train_set = j[0] + j[1] + j[2] + j[3]
# f.close()
# train_datas = MyDataset(
#     r"C:\Users\BOURNE\Desktop\zzn_project\Bundle_Recommendation\dataset\train_dataset", train_set)
# test_set = j[4]
# test_datas = MyDataset(
#     r"C:\Users\BOURNE\Desktop\zzn_project\Bundle_Recommendation\dataset\test_dataset", test_set)
# model = BasicModel(train_datas)
# optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
# train_process(train_datas, test_datas, model, 100, 1, 3e-5, 2e-3, 100, 2e-7)
# prepare datas to train





# f = open(r"C:\Users\Administrator\Desktop\SGCN\baseline_data\Youshu\current_usage\1\train_datas", "rb")
# datas = pickle.load(f)
# f.close()
# f = open(r"C:\Users\Administrator\Desktop\SGCN\baseline_data\Youshu\current_usage\1\train_labels", "rb")
# labels = pickle.load(f)
# f.close()
# f = open(r"C:\Users\Administrator\Desktop\SGCN\baseline_data\Youshu\current_usage\1\train_pair", "rb")
# pairs = pickle.load(f)
# f.close()
# poss,negs = prepare_bpr_dataset_dict(datas,labels,pairs)
# n_p = []
# n_n = []
# for i in range(len(poss)):
#     n_n.append(negs[i][1])
#     n_p.append(poss[i][1])
# num = len(n_n)
# train_np = n_p[:int(num*0.7)]
# train_nn = n_n[:int(num*0.7)]
# test_set = n_p[int(num*0.7):] + n_n[int(num*0.7):]
# # prepare dataset
# train_datasp = MyDataset(
#     r"C:\Users\Administrator\Desktop\SGCN\baseline_data\Youshu\torch_mydataset\1\train_dataset_BPR_pos", train_np)
# torch.save(train_datasp,"train_datasp.pth")
# train_datasn = MyDataset(
#     r"C:\Users\Administrator\Desktop\SGCN\baseline_data\Youshu\torch_mydataset\1\train_dataset_BPR_neg", train_nn)
# torch.save(train_datasn, "train_datasn.pth")
# test_datas = MyDataset(
#     r"C:\Users\Administrator\Desktop\SGCN\baseline_data\Youshu\torch_mydataset\1\test_dataset_BPR", test_set)
# torch.save(test_datas, "test_datas.pth")
# train_datasp = torch.load(r"C:\Users\Administrator\Desktop\SGCN\train_datasp.pth")
# train_datasn = torch.load(r"C:\Users\Administrator\Desktop\SGCN\train_datasn.pth")
# test_datas = torch.load(r"C:\Users\Administrator\Desktop\SGCN\test_datas.pth")
# model = BasicModel(train_datasp)
# optimizer = Adam(model.parameters(), lr=3e-5, weight_decay=1e-5)
#
#
# train_process(train_data,test_datas,model, 100, 1, 3e-5, 2e-3, 100, 2e-6)








# cur_rmse, cur_bad = get_rmse(
#     model, test_loader, device, show_progress=1)
#
# print(cur_rmse)

#------------------------------TEST--------------------------------



#-----------------------------TRAIN---------------------------------
train_datasp = torch.load(
    r"C:\Users\Administrator\Desktop\SGCN\train_datasp.pth")
train_datasn = torch.load(
    r"C:\Users\Administrator\Desktop\SGCN\train_datasn.pth")

model = BasicModel(train_datasp)
optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

test_datas = torch.load(
    r"C:\Users\Administrator\Desktop\SGCN\test_datas.pth")

train_process(train_datasp, train_datasn, test_datas,model, 10, 1, 3e-5, 2e-3, 100, 2e-6)


#---------------------------PREPARE DATA----------------------------
# f = open(r"C:\Users\BOURNE\Desktop\zzn_project\Bundle_Recommendation\dataset\current_usage_BPR\train_datas_181_192", "rb")
# datas = pickle.load(f)
# f.close()
# f = open(r"C:\Users\BOURNE\Desktop\zzn_project\Bundle_Recommendation\dataset\current_usage_BPR\train_labels_181_192", "rb")
# labels = pickle.load(f)
# f.close()
# f = open(r"C:\Users\BOURNE\Desktop\zzn_project\Bundle_Recommendation\dataset\current_usage_BPR\train_pairs_181_192", "rb")
# pairs = pickle.load(f)
# f.close()
# poss,negs = prepare_bpr_dataset_dict(datas,labels,pairs)
# n_p = []
# n_n = []
# for i in range(len(poss)):
#     n_n.append(negs[i][1])
#     n_p.append(poss[i][1])
# num = len(n_n)
# train_np = n_p[:int(num*0.7)]
# train_nn = n_n[:int(num*0.7)]
# test_set = n_p[int(num*0.7):] + n_n[int(num*0.7):]
# # prepare dataset
# train_datasp = MyDataset(
#     r"C:\Users\BOURNE\Desktop\zzn_project\Bundle_Recommendation\dataset\train_dataset_BPR_pos_181_192", train_np)
# torch.save(train_datasp,"train_datasp_181_192.pth")
# train_datasn = MyDataset(
#     r"C:\Users\BOURNE\Desktop\zzn_project\Bundle_Recommendation\dataset\train_dataset_BPR_neg_181_192", train_nn)
# torch.save(train_datasn, "train_datasn_181_192.pth")
# test_datas = MyDataset(
#     r"C:\Users\BOURNE\Desktop\zzn_project\Bundle_Recommendation\dataset\test_dataset_BPR_181_192", test_set)
# torch.save(test_datas, "test_datas_181_192.pth")
