from model import BasicModel
import os
import time
import numpy as np
import math
import multiprocessing as mp
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import sys
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data, Dataset, DataLoader, DenseDataLoader, InMemoryDataset
from tqdm import tqdm
from glob import glob
import torch_geometric.data.storage
sys.path.extend([r"E:\SGCN\utils"])
from baseline_preprocess_Youshu import *
from render_dataset import *

# select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

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


# compute loss and rmse
def eval_loss(model, loader, device, regression=False, show_progress=False):
    # model.eval()
    loss = 0
    bad = 0
    # if show_progress:
    #     # print('Testing begins...')
    #     pbar = tqdm(loader)
    # else:
    #     pbar = loader
    for data in loader:
        data = data.to(device)

        out = model(data)
        # print(out,data.y)
        if out == None:
            bad += 1
            continue
        if regression:
            loss += F.mse_loss(out, data.y.view(-1), reduction='sum').item()
        else:
            loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
        torch.cuda.empty_cache()
    return loss / (len(loader.dataset) - bad),bad

def eval_loss_test(model, loader, device, regression=False, show_progress=False):
    results = []
    model.eval()
    loss = 0
    bad = 0
    pos_loss = 0
    pos_num = 0
    neg_num = 0
    neg_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)


            out = model(data)
            # print(out,data.y)
            results.append(out)
            if out == None:
                bad += 1
                continue

            k = F.mse_loss(out, data.y.view(-1), reduction='sum').item()
            if data.y.cpu().numpy() == 1:
                pos_loss += k
                pos_num += 1
            else:
                neg_loss += k
                neg_num += 1
            loss += k
            torch.cuda.empty_cache()
    return loss / (len(loader.dataset) - bad), pos_loss/pos_num,neg_loss/neg_num,results

def get_rmse(model, loader, device, show_progress=False):
    mse_loss,bad = eval_loss(model, loader, device, True, show_progress)
    rmse = math.sqrt(mse_loss)
    return rmse,bad

def get_rmse_test(model, loader, device, show_progress=False):
    mse_loss,pos_loss,neg_loss,result= eval_loss_test(model, loader, device, True, show_progress)
    rmse = math.sqrt(mse_loss)
    return rmse,math.sqrt(pos_loss),math.sqrt(neg_loss),result
# define train once process
def train(model,
          optimizer,
          dataloader,
          device,
          regression=True,
          ARR=0,
          show_progress=True,
          epoch=None):

    # open batch normalization
    model.train()
    total_loss = 0

    # show progress or not
    # pbar = tqdm(dataloader)

    bad = 0
    i = 0
    print("---Start Training---")
    for databatch in dataloader:
        i += 1
        # databatch.to(device)
        optimizer.zero_grad()

        # randomnize the input features
        # new_x = torch.FloatTensor(databatch.x.size()[0],databatch.num_features)
        # nn.init.xavier_normal_(new_x)
        # databatch.x = new_x
        databatch = databatch.to(device)

        result = model(databatch)

        if result is None:
            bad += 1
            continue

        if regression:
            cur_loss = F.mse_loss(result,databatch.y.view(-1))
        else:
            cur_loss = F.nll_loss(result,databatch.y.view(-1))
        


        cur_loss.backward()

        if databatch.batch is not None:
            num_graph = databatch.num_graphs
        else:
            num_graph = databatch.x.size(0)

        total_loss += cur_loss.item()

        optimizer.step()
        torch.cuda.empty_cache()
    print("The bad training batch is: ",bad)
    return total_loss / len(dataloader.dataset)

    


# define whole train process
def train_process(model_name,
                  train_dataset,
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
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size,shuffle=False)   # do not need to shuffle test data
    
    # prepare model
    model = model.to(device)
    model.reset_parameters()
    optimizer = Adam(model.parameters(),lr=lr,weight_decay=weight_decay)

    cur_epoch = 1

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    
    if show_progress:
        pbar = tqdm(range(cur_epoch,epochs + cur_epoch))
    else:
        pbar = range(cur_epoch,epochs + cur_epoch)

    rmses = []

    start_time = time.perf_counter()


    for epoch in pbar:
        trainloss = train(model,optimizer,train_loader,device,regression=True,ARR=ARR,show_progress=show_progress,epoch=epoch)
        cur_bad = 0
        cur_rmse, cur_bad, results = get_rmse_test(model, test_loader, device,show_progress=show_progress)

        if show_progress:
            print(
                'Epoch {}, train loss {:.6f},cur test error {:.6f}'.format(
                    epoch,trainloss,cur_rmse)
            )
        
        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']
    
    torch.save(model,model_name)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    cur_rmse, cur_bad, results = get_rmse_test(model, test_loader, device, show_progress=show_progress)
    end_time = time.perf_counter()

    time_interval = end_time - start_time

    print('Test RMSE:  {:.6f}, Duration: {:.6f}'.format(cur_rmse,time_interval))






# define whole train process
def train_process_sgd(train_dataset,
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)  # do not need to shuffle test data

    # prepare model
    model = model.to(device)
    model.reset_parameters()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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
        trainloss = train(model, optimizer, train_loader, device, regression=True, ARR=ARR, show_progress=show_progress,
                          epoch=epoch)
        cur_bad = 0
        cur_rmse, cur_bad = get_rmse(
            model, test_loader, device, show_progress=show_progress)

        if show_progress:
            print(
                'Epoch {}, train loss {:.6f},cur test error {:.6f}'.format(
                    epoch, trainloss, cur_rmse)
            )
        for i in model.parameters():
            print(i)
        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

    torch.save(model, "my_model_4.pth")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    cur_rmse, cur_bad = get_rmse(
        model, test_loader, device, show_progress=show_progress)
    end_time = time.perf_counter()

    time_interval = end_time - start_time

    print('Test RMSE:  {:.6f}, Duration: {:.6f}'.format(cur_rmse, time_interval))


def test_process(test_dataset, model, device, batch_size=1):
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    model.to(device)
    model.eval()

    with torch.no_grad():
        loss = get_rmse(model, test_loader, device)

    print('Test RMSE:  {:.6f}'.format(loss[0]))


# train_datas = load_ob j(

#     r"C:\Users\BOURNE\Desktop\zzn_project\SGCN\baseline_data\Youshu\train_test_data\4\Mydatasets\train\train_balanced.pkl")

# test_datas = load_obj(
#     r"C:\Users\BOURNE\Desktop\zzn_project\SGCN\baseline_data\Youshu\train_test_data\4\Mydatasets\test\test_balanced.pkl")

# model = BasicModel(train_datas)
# # training
# train_process("My_model_MSE_NPratio2_allYoushu_12epoch.pth", train_datas,
#               test_datas, model, 12, 1, 3e-5, 2e-3, 100, 2e-7)

# # test for rmse of 0.75 drop
# model = torch.load(r"E:\SGCN\models\My_model_MSE_NPratio2_allYoushu_4epoch.pth")
# #
# #
# # print("---0.75 drop starts---")
# #
# # test_data = load_obj(r"E:\SGCN\test_data_stage2_drop0.75_0_300.pkl")
# #
# # test_set = MyDataset(r"E:\SGCN\A",test_data)
# # a = get_rmse_test(model, DataLoader(test_set,shuffle=False,batch_size=1), device, show_progress=False)
# # print("0-300 users",a[0],a[1],a[2])
# # del test_data
# # del test_set
# #
# # test_data = load_obj(r"E:\SGCN\test_data_stage2_drop0.75_300_600.pkl")
# # test_set = MyDataset(r"E:\SGCN\B",test_data)
# # a = get_rmse_test(model, DataLoader(test_set,shuffle=False,batch_size=1), device, show_progress=False)
# # print("300-600 users",a[0],a[1],a[2])
# # del test_data
# # del test_set
# #
# # test_data = load_obj(r"E:\SGCN\test_data_stage2_drop0.75_600_900.pkl")
# # test_set = MyDataset(r"E:\SGCN\C",test_data)
# # a = get_rmse_test(model, DataLoader(test_set,shuffle=False,batch_size=1), device, show_progress=False)
# # print("600-900 users",a[0],a[1],a[2])
# # del test_data
# # del test_set
# #
# # test_data = load_obj(r"E:\SGCN\test_data_stage2_drop0.75_900_1200.pkl")
# # test_set = MyDataset(r"E:\SGCN\D",test_data)
# # a = get_rmse_test(model, DataLoader(test_set,shuffle=False,batch_size=1), device, show_progress=False)
# # print("900-1200 users",a[0],a[1],a[2])
# # del test_data
# # del test_set
# #
# # print("---drop0.75 finish---")
# #
# # print()
# # print()
# #
# print()
# print()
# print("---0.5 drop starts---")
#
# test_data = load_obj(r"E:\SGCN\test_data_stage2_drop0.5_0_300.pkl")
# test_set = MyDataset(r"E:\SGCN\A1",test_data)
# a = get_rmse_test(model, DataLoader(test_set,shuffle=False,batch_size=1), device, show_progress=False)
# result = a[3]
# save_obj(result,r"E:\SGCN\drop0.5_result_user0-300.pkl")
# print("0-300 users",a[0],a[1],a[2])
# del test_data
# del test_set
#
# test_data = load_obj(r"E:\SGCN\test_data_stage2_drop0.5_300_600.pkl")
# test_set = MyDataset(r"E:\SGCN\B1",test_data)
# a = get_rmse_test(model, DataLoader(test_set,shuffle=False,batch_size=1), device, show_progress=False)
# result = a[3]
# save_obj(result,r"E:\SGCN\drop0.5_result_user300-600.pkl")
# print("300-600 users",a[0],a[1],a[2])
# del test_data
# del test_set
#
# test_data = load_obj(r"E:\SGCN\test_data_stage2_drop0.5_600_900.pkl")
# test_set = MyDataset(r"E:\SGCN\C1",test_data)
# a = get_rmse_test(model, DataLoader(test_set,shuffle=False,batch_size=1), device, show_progress=False)
# result = a[3]
# save_obj(result,r"E:\SGCN\drop0.5_result_user600-900.pkl")
# print("600-900 users",a[0],a[1],a[2])
# del test_data
# del test_set
#
# test_data = load_obj(r"E:\SGCN\test_data_stage2_drop0.5_900_1200.pkl")
# test_set = MyDataset(r"E:\SGCN\D1",test_data)
# a = get_rmse_test(model, DataLoader(test_set,shuffle=False,batch_size=1), device, show_progress=False)
# result = a[3]
# save_obj(result,r"E:\SGCN\drop0.5_result_user900-1200.pkl")
# print("900-1200 users",a[0],a[1],a[2])
# del test_data
# del test_set
#
# print("---drop0.5 finish---")
#
# print()
# print()
#
#
# print("---0.25 drop starts---")
#
# test_data = load_obj(r"E:\SGCN\test_data_stage2_drop0.25_0_300.pkl")
# test_set = MyDataset(r"E:\SGCN\A2",test_data)
# a = get_rmse_test(model, DataLoader(test_set,shuffle=False,batch_size=1), device, show_progress=False)
# result = a[3]
# save_obj(result,r"E:\SGCN\drop0.25_result_user0-300.pkl")
# print("0-300 users",a[0],a[1],a[2])
# del test_data
# del test_set
#
# test_data = load_obj(r"E:\SGCN\test_data_stage2_drop0.25_300_600.pkl")
# test_set = MyDataset(r"E:\SGCN\B2",test_data)
# a = get_rmse_test(model, DataLoader(test_set,shuffle=False,batch_size=1), device, show_progress=False)
# result = a[3]
# save_obj(result,r"E:\SGCN\drop0.25_result_user300-600.pkl")
# print("300-600 users",a[0],a[1],a[2])
# del test_data
# del test_set
#
# test_data = load_obj(r"E:\SGCN\test_data_stage2_drop0.25_600_900.pkl")
# test_set = MyDataset(r"E:\SGCN\C2",test_data)
# a = get_rmse_test(model, DataLoader(test_set,shuffle=False,batch_size=1), device, show_progress=False)
# result = a[3]
# save_obj(result,r"E:\SGCN\drop0.25_result_user600-900.pkl")
# print("600-900 users",a[0],a[1],a[2])
# del test_data
# del test_set
#
# test_data = load_obj(r"E:\SGCN\test_data_stage2_drop0.25_900_1200.pkl")
# test_set = MyDataset(r"E:\SGCN\D2",test_data)
# a = get_rmse_test(model, DataLoader(test_set,shuffle=False,batch_size=1), device, show_progress=False)
# result = a[3]
# save_obj(result,r"E:\SGCN\drop0.25_result_user900-1200.pkl")
# print("900-1200 users",a[0],a[1],a[2])
# del test_data
# del test_set
#
# print("---drop0.25 finish---")