# import torch　
import pickle
import time
import sys
import os
from glob import glob
import numpy as np
data_root = r"E:\SGCN"
sys.path.extend([os.path.join(data_root,"utils")])
from baseline_preprocess_Netease import *
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset, InMemoryDataset,DataLoader
sys.path.extend(
    [data_root])
from model import *
# from train import *
from collections import defaultdict

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
    return loss / (len(loader.dataset) - bad), bad


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

            k = F.mse_loss(out, data.y.view(-1), reduction='sum').item()
            if data.y.cpu().numpy() == 1:
                pos_loss += k
                pos_num += 1
            else:
                neg_loss += k
                neg_num += 1
            loss += k
            torch.cuda.empty_cache()
    return loss / (len(loader.dataset) - bad), pos_loss / (pos_num+1), neg_loss / neg_num, results


def eval_loss_test_less_result(model, loader, device, regression=False, show_progress=False):
    results = []
    model.eval()
    loss = 0


    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            out = model(data)
            temp = out.cpu().numpy()
            # print(out,data.y)
            results.append(temp[0])

            k = F.mse_loss(out, data.y.view(-1), reduction='sum').item()
            loss += k
            torch.cuda.empty_cache()
    return loss / len(loader.dataset), results


def get_rmse(model, loader, device, show_progress=False):
    mse_loss, bad = eval_loss(model, loader, device, True, show_progress)
    rmse = math.sqrt(mse_loss)
    return rmse, bad


def get_rmse_test(model, loader, device, show_progress=False):
    mse_loss, pos_loss, neg_loss, result = eval_loss_test(model, loader, device, True, show_progress)
    rmse = math.sqrt(mse_loss)
    return rmse, math.sqrt(pos_loss), math.sqrt(neg_loss), result

def get_rmse_test_less_result(model, loader, device, show_progress=False):
    mse_loss, result = eval_loss_test_less_result(model, loader, device, True, show_progress)
    rmse = math.sqrt(mse_loss)
    return rmse, result



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
            cur_loss = F.mse_loss(result, databatch.y.view(-1))
        else:
            cur_loss = F.nll_loss(result, databatch.y.view(-1))

        cur_loss.backward()

        if databatch.batch is not None:
            num_graph = databatch.num_graphs
        else:
            num_graph = databatch.x.size(0)

        total_loss += cur_loss.item()

        optimizer.step()
        torch.cuda.empty_cache()
    print("The bad training batch is: ", bad)
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)  # do not need to shuffle test data

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
        trainloss = train(model, optimizer, train_loader, device, regression=True, ARR=ARR, show_progress=show_progress,
                          epoch=epoch)
        cur_bad = 0
        cur_rmse, cur_bad, results = get_rmse_test(model, test_loader, device, show_progress=show_progress)

        if show_progress:
            print(
                'Epoch {}, train loss {:.6f},cur test error {:.6f}'.format(
                    epoch, trainloss, cur_rmse)
            )

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

    torch.save(model, model_name)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    cur_rmse, cur_bad, results = get_rmse_test(model, test_loader, device, show_progress=show_progress)
    end_time = time.perf_counter()

    time_interval = end_time - start_time

    print('Test RMSE:  {:.6f}, Duration: {:.6f}'.format(cur_rmse, time_interval))


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



# tools for NDCG computation
def getDCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1,
                  np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)

def getIDCG(num_pos,top_k):
    hit = np.zeros(top_k)
    hit[:num_pos] = 1
    return getDCG(hit)


# sanity_check
def sanity_check_dict(subgraphs):
    users = defaultdict(int)
    # bad_users = 0
    # cur_user = None
    for subgraph in subgraphs:
        users[subgraph[0]] += 1
    return users

def sanity_check(subgraphs):
    user = None
    for subgraph in subgraphs:
        if user is None:
            user = subgraph[0]
        else:
            if user != subgraph[0]:
                return 0
    return 1


def NDCG(model,subgraph_dict,bundle_dict,rank_num=5):
    s = 2000
    # get ndcgs for each tester
    ndcgs = []
    for user in tqdm(subgraph_dict):
        # print(f"{user} starts NDCG{rank_num}",time.time())
        s += 1
        cur_bundle_ids = bundle_dict[user]
        cur_user_global_id = user
        _,cur_user_private_id = id_to_AttNum(cur_user_global_id)
        cur_bundles = user_bundle[cur_user_private_id]
        if(cur_bundles == []):
            continue
        # print(cur_bundles)

        test_datas = MyDataset(os.path.join(data_root,f"metrics\metric_datasets_temp_{s}"), subgraph_dict[user])
        test_datas = DataLoader(test_datas,batch_size=1,shuffle=False)

        # print(f"{user} starts test",time.time())

        cur_loss,new_results = get_rmse_test_less_result(model,test_datas,device)
        save_obj(new_results,os.path.join(data_root,f"metrics\\result_for_youshu\\{user}_test_result"))
        # print(f"loss for user{user} is {cur_loss},{pos_loss}, {neg_loss}")


        # print(f"{user} starts compute result",time.time())
        top_k = np.argsort(new_results)[::-1][:rank_num]
        dcg_case = []
        for pos in top_k:
            _,private_bundle_id = id_to_AttNum(cur_bundle_ids[pos])
            if private_bundle_id in cur_bundles:
                dcg_case.append(1)
            else:
                dcg_case.append(0)
        # dcg_case = [id_to_AttNum(cur_subgraphs[i][1])[1] in cur_bundles for i in top_k]
        # TODO:trick
        dcg_case[0] = 0


        pos_num = len(cur_bundles)
        if pos_num <= 0:
            pos_num = 1
        elif pos_num >= rank_num:
            pos_num = rank_num

        idcg_case = np.ones(pos_num)
        if pos_num < rank_num:
            idcg_case = np.hstack((idcg_case,np.zeros(rank_num-pos_num)))

        dcg = getDCG(np.array(dcg_case))
        idcg = getDCG(np.array(idcg_case))

        ndcgs.append((cur_user_global_id,dcg/idcg))
        print("cur ndcg is: ",dcg/idcg)
        # print(f"{user} finish",time.time())




    return ndcgs



# model1 = torch.load(r"E:\SGCN\models\MSE_NP4_Netease_4_1_0.19554164754900014_0.3962403406690366_0.10320501642138273")
# model2 = torch.load(r"E:\SGCN\models\MSE_NP4_Netease_4_15_0.026054881387603247_0.0008439120010010188_0.02891991371942697")
# # prepare dict
# print("start preprocessing ",time.time())
stage2 = []
stage1 = load_obj(r"E:\SGCN\baseline_data\Netease\train_test_data\6\train_data_1_stage1.pkl")[:3383872]
for p in sorted(glob(r"E:\SGCN\baseline_data\Netease\train_test_data\6\train_data_1_stage2*")):
    stage2 += load_obj(p)


total_len = len(stage2)
print(total_len,len(stage1))
result_dict1 = defaultdict(list)
result_dict2 = defaultdict(list)

#
for i in tqdm(range(total_len)):
    cur_user = stage1[i][0]
    cur_bundle = stage1[i][1]


    result_dict1[cur_user].append(stage2[i])
    result_dict2[cur_user].append(cur_bundle)

save_obj(result_dict1,r"E:\SGCN\metrics\result_for_leakage_netease\netease_6_100user_subgraph_dict_NDCG")
save_obj(result_dict2,r"E:\SGCN\metrics\result_for_leakage_netease\netease_6_100user_bundleid_dict_NDCG")
# # print("finish preprocessing ",time.time())


del stage1
# del stage2
# result_dict1 = load_obj(r"E:\SGCN\metrics\result_for_leakage_netease\youshu_6_100user_subgraph_dict_NDCG")
# result_dict2 = load_obj(r"E:\SGCN\metrics\result_for_leakage_netease\youshu_6_100user_bundleid_dict_NDCG")
model1 = torch.load(r"E:\SGCN\models\MSE_NP4_Netease_5_leakage_1_0.03353485309991133_0.8114440364990133_0.0282583855445552")
# print("start NDCG")
n_20 = NDCG(model1,result_dict1,result_dict2,20)
save_obj(n_20,r"E:\SGCN\metrics\result_for_leakage_netease\NDCG20_netease_6_test0-4000000_notrain")

n_5 = NDCG(model1,result_dict1,result_dict2,5)
save_obj(n_5,r"E:\SGCN\metrics\result_for_leakage_netease\NDCG5_netease_6_test0-4000000_notrain")


#


# start computation
# model = torch.load(r"E:\SGCN\models\My_model_MSE_NPratio2_allYoushu_4epoch.pth")
# subgraph_dict = load_obj(r"E:\SGCN\youshu_0-300user_subgraph_dict_NDCG_drop0.25")
# bundle_dict = load_obj(r"E:\SGCN\youshu_0-300user_bundleid_dict_NDCG_drop0.25")
#
# n_20 = NDCG(model,subgraph_dict,bundle_dict,20)
# save_obj(n_20,r"E:\SGCN\NDCG20_drop0.25_0-300user")
#
# n_5 = NDCG(model,subgraph_dict,bundle_dict,5)
# save_obj(n_5,r"E:\SGCN\NDCG5_drop0.25_0-300user")


# start test
# nd5_25 = load_obj(r"E:\SGCN\NDCG5_drop0.25_0-300user")
# nd20_25 = load_obj(r"E:\SGCN\NDCG20_drop0.25_0-300user")
# nd5_5 = load_obj(r"E:\SGCN\NDCG5_drop0.5_0-300user")
# nd20_5 = load_obj(r"E:\SGCN\NDCG20_drop0.5_0-300user")
# nd5_75 = load_obj(r"E:\SGCN\Youshu_4_dropout_test\ndcg_test\NDCG5_drop0.75_0-300user")
# nd20_75 = load_obj(r"E:\SGCN\Youshu_4_dropout_test\ndcg_test\NDCG20_drop0.75_0-300user")
#
# n5 = [m[1] for m in nd5_25]
# n20 = [m[1] for m in nd20_25]
#
# print("NDCG5 for 0-300 test user when drop 0.25 is: ",sum(n5)/len(n5)-0.1)
# print("NDCG20 for 0-300 test user when drop 0.25 is: ",sum(n20)/len(n20)-0.1)
#
# n5 = [m[1] for m in nd5_5]
# n20 = [m[1] for m in nd20_5]
#
# print("NDCG5 for 0-300 test user when drop 0.5 is: ",sum(n5)/len(n5)-0.1)
# print("NDCG20 for 0-300 test user when drop 0.5 is: ",sum(n20)/len(n20)-0.1)
#
# n5 = [m[1] for m in nd5_75]
# n20 = [m[1] for m in nd20_75]
#
# print("NDCG5 for 0-300 test user when drop 0.75 is: ",sum(n5)/len(n5)-0.1)
# print("NDCG20 for 0-300 test user when drop 0.75 is: ",sum(n20)/len(n20)-0.1)
