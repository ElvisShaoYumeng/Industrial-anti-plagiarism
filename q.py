
# 数据分析
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 通用
import random
import time
from datetime import datetime
import copy
import argparse
from tqdm import tqdm
import os
from pathlib import Path

# Pytorch
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils import data
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# 读取数据集
from utils.helper_CWRU import get_df_all, download

#网络
from utils.train_utils import accuracy, average_weights, average_compt_weights
from utils.init_utils import seed_torch
from utils.maml_model import Net4CNN, Net4CNN_add, Net4CNN_PU, BP
#import learn2learn as l2l
import optuna
import plotly
from Noise_add import noise_add1



x = np.load('./Data/CWRU/generated/x_c0.npy')
print(x.shape)
argparser = argparse.ArgumentParser()

#数据
argparser.add_argument('--signal_length', type=int, help='signal_length', default=1024)
argparser.add_argument('--ways', type=int, help='n way', default=10)
#argparser.add_argument('--shots', type=int, help=' ', default=5)
#argparser.add_argument('--train_task_num', type=int, help=' ', default=10000)
#argparser.add_argument('--valid_task_num', type=int, help=' ', default=10000)
#argparser.add_argument('--test_task_num',  type=int, help=' ', default=10000)
#argparser.add_argument('--sample_time',  type=int, help=' ', default=100)

#其他
argparser.add_argument('--device', type=int, help='n way', default=torch.device('cpu'))  #torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

#超参数
argparser.add_argument('--meta_lr', type=float, help=' ', default=0.01)
argparser.add_argument('--fast_lr', type=float, help=' ', default=0.01)
argparser.add_argument('--local_lr', type=float, help=' ', default=0.001)
argparser.add_argument('--epoch', type=int, help=' ', default=200)
argparser.add_argument('--train_meta_batch_size', type=int, help=' ', default=14)  #训练用户数量
argparser.add_argument('--valid_meta_batch_size', type=int, help=' ', default=3)
argparser.add_argument('--test_meta_batch_size' , type=int, help=' ', default=14)  #测试用户数量
argparser.add_argument('--local_epoch' , type=int, help=' ', default=3)
#argparser.add_argument('--local_batch_size' , type=int, help=' ', default=512)

args = argparser.parse_args(args = [])

class Dataset_CWRU(data.Dataset):
    def __init__(self, args, condition, mode, location):
        super().__init__()
        self.sample_len = 1024
        # self.task_mode = True if ways == 10 else False
        self.__getdata__(mode, condition, location)
        print('condition',condition)
        print('location', location)
    def __getdata__(self, mode, condition, location):

        # 工况 K0 下的 10 分类   [类别，每一类取多少个，样本数]

        x = np.load('./Data/CWRU/generated/x_c' + str(condition) + '.npy')
        y = np.load('./Data/CWRU/generated/y_c' + str(condition) + '.npy')

        if mode == 'train':
            #20&18
            # x = x[:, 0 + 20*location:16 + 20*location, :]
            # y = y[:, 0 + 20*location:16 + 20*location]
            #16&14
            x = x[:, 0 + 25 * location:20 + 25 * location, :]
            y = y[:, 0 + 25 * location:20 + 25 * location]
        elif mode == 'test':
            #20&18
            # x = x[:, 16 + 20*location:20 + 20*location, :]
            # y = y[:, 16 + 20*location:20 + 20*location]
            #16&14
            x = x[:, 20 + 25*location:25 + 25*location, :]
            y = y[:, 20 + 25*location:25 + 25*location]


        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.x = self.x.reshape([-1, 1, self.sample_len])  # x: (n_way*n, len, 1), y: (n_way*n)
        self.y = self.y.reshape(-1)

    def __getitem__(self, item):
        x = self.x[item]  # (NC, l)
        y = self.y[item]
        return x, y  # , label

    def __len__(self):
        return len(self.x)


args.gpu = -1  ###cpu -1 gpu 1
batch_size = 256
optimizer_name = "Adam"  ##
lr = 0.0001  ##
lazy_ratio = 0  ## 抄袭比例
num_lazy = int(lazy_ratio * args.train_meta_batch_size)
noise_scale = 0.0015
num_experiments = 10


class FedAvg_learner(object):
    def __init__(self, args):
        self.args = args;
        h_size = 64;
        layers = 4;
        sample_len = 1024;
        feat_size = (sample_len // 2 ** layers) * h_size
        self.model = Net4CNN_PU(output_size=self.args.ways, hidden_size=h_size, layers=layers, channels=1,
                                embedding_size=feat_size).to(self.args.device).double()
        # self.model = BP().to(self.args.device).double()

    def train(self, model, train_loader):
        opt = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)  ##
        critiren = torch.nn.CrossEntropyLoss(reduction='mean')
        for local_epoch in range(self.args.local_epoch):
            correct, total = 0, 0
            for x, y in train_loader:
                x = x.to(args.device);
                y = y.to(args.device)
                out = model(x)
                loss = critiren(out, y.long())
                opt.zero_grad()
                loss.backward()
                opt.step()
                correct += torch.eq(torch.argmax(out, dim=1), y).sum()
                total += y.shape[0]
        return model, torch.true_divide(correct, total).item()

    def valid(self, model, valid_loader):
        with torch.no_grad():
            correct, total = 0, 0
            for x, y in valid_loader:
                x = x.to(args.device);
                y = y.to(args.device)
                out = model(x)
                correct += torch.eq(torch.argmax(out, dim=1), y).sum()
                total += y.shape[0]
        return torch.true_divide(correct, total).item()

    def test(self, model, test_loader):
        with torch.no_grad():
            correct, total = 0, 0
            for x, y in test_loader:
                x = x.to(args.device);
                y = y.to(args.device)
                out = model(x)
                correct += torch.eq(torch.argmax(out, dim=1), y).sum()
                total += y.shape[0]
        return torch.true_divide(correct, total).item()


#### start ####

path = os.path.join(os.getcwd(), 'Saved_models_CWRU_11.09', 'FedAvg',
                    datetime.strftime(datetime.now(), '%m%d-%H%M%S'));
save_path = path
Net = FedAvg_learner(args)

# 数据
train_loaders = {}

test_loaders = {}
# 20&18
# conditions = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3]
# locations = [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]
#16&14
conditions = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
locations = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
for client in range(Net.args.train_meta_batch_size):
    train_loaders[client] = torch.utils.data.DataLoader(Dataset_CWRU(Net.args, condition=conditions[client], mode='train', location= locations[client]),
                                                        batch_size=batch_size, shuffle=True)

for client in range(Net.args.train_meta_batch_size):
    test_loaders[client] = torch.utils.data.DataLoader(Dataset_CWRU(Net.args, condition=conditions[client], mode='test', location= locations[client]),
                                                       batch_size=batch_size, shuffle=True)

best_test = 0  # 用于保存最佳模型
exp_train_acc = []
exp_test_acc = []
final_test_accuracy = []
# 模型
model = Net.model
w_init = copy.deepcopy(model.state_dict())

for exp_time in range(num_experiments):
    train_accuracy = []
    test_accuracy = []  # 保存每个epoch的准确率

    for epoch in range(Net.args.epoch):
        t0 = time.time()
        train_acc_in_all_client = 0;

        test_acc_in_all_client = 0

        w_locals = []
        for client in range(Net.args.train_meta_batch_size):
            if epoch == 0:
                model.load_state_dict(w_init)
            else:
                model.load_state_dict(w_glob)
            if client < (Net.args.train_meta_batch_size - num_lazy):
                model.train()
                model, train_acc = Net.train(model, train_loaders[client])
                train_acc_in_all_client += train_acc
                w_locals.append(copy.deepcopy(model.state_dict()))
            else:
                ###  copy  ###
                k = random.randint(0, (client - 1))
                lazy_locals = copy.deepcopy(w_locals[k])
                lazy_locals = noise_add1(args, noise_scale, lazy_locals)
                w_locals.append(copy.deepcopy(lazy_locals))
                #print(k)

        ### 不检测
        rho = [1 / args.train_meta_batch_size] * args.train_meta_batch_size
        ### 抄袭检测
        # rho1 = [1 / (args.train_meta_batch_size - num_lazy)] * (args.train_meta_batch_size - num_lazy)
        # rho2 = [0] * num_lazy
        # rho = rho1 + rho2
        w_glob = average_compt_weights(w_locals,rho)

        ###  基于贡献度的聚合  ###
        ### 计算各用户贡献度
        # contri_list = []
        # for contri_idx in range(Net.args.train_meta_batch_size - rho.count(0)):
        #     temp_rho = copy.deepcopy(rho)
        #     temp_rho[contri_idx] = 0  ### 生成排除单个用户的全局模型
        #     for idx in range(len(temp_rho)):
        #         if temp_rho[idx] > 0:
        #             temp_rho[idx] = 1 / (Net.args.train_meta_batch_size - temp_rho.count(0))
        #     w_exg = average_compt_weights(w_locals,temp_rho)
        #     diff_list = []
        #     for client in range(Net.args.train_meta_batch_size): ###在其他用户端测试精度
        #         model.load_state_dict(w_glob)
        #         model.eval()
        #         acc_contri = Net.test(model, test_loaders[client])
        #         model.load_state_dict(w_exg)
        #         model.eval()
        #         acc_contri_idx = Net.test(model, test_loaders[client])
        #         diff = acc_contri - acc_contri_idx
        #         diff_list.append(diff)
        #     diff_idx = sum(diff_list) / len(diff_list)
        #     contri_list.append(diff_idx)
        #
        # ### 由贡献度获得权重
        # for idx in range(len(contri_list)):  ###贡献度微调
        #     if contri_list[idx] < 0:
        #         contri_list[idx] = 0
        # if all(item == 0 for item in contri_list) == False:
        #     for idx in range(len(contri_list)):
        #         if contri_list == 0:
        #             rho[idx] = 0
        #         else:
        #             rho[idx] = contri_list[idx] / sum(contri_list)
        # w_glob = average_compt_weights(w_locals, rho)  ### 根据贡献度得到的全局模型

        ### 每个epoch最后的精度测试
        for client in range(Net.args.train_meta_batch_size):
            model.load_state_dict(w_glob)
            model.eval()
            test_acc = Net.test(model, test_loaders[client])
            test_acc_in_all_client += test_acc

        if test_acc_in_all_client / Net.args.train_meta_batch_size > best_test:
            best_test = test_acc_in_all_client / Net.args.train_meta_batch_size
            torch.save(model.state_dict(), save_path)

        train_accuracy.append(train_acc_in_all_client / Net.args.train_meta_batch_size)
        test_accuracy.append(test_acc_in_all_client / Net.args.train_meta_batch_size)
        ACCURACY = test_acc_in_all_client / Net.args.train_meta_batch_size

            # 打印参数
        t1 = time.time()
        if epoch % 20 == 0:
            print(f'Time /epoch: {t1 - t0:.4f} s', end='       ');
            print('Epoch', epoch + 1, end='       ')
            print(f'Train Accuracy:', round(train_acc_in_all_client / Net.args.train_meta_batch_size, 4), end='   ')
            print(f'Test  Accuracy:', round(test_acc_in_all_client / Net.args.train_meta_batch_size, 4))

    exp_train_acc.append(train_accuracy)
    exp_test_acc.append(test_accuracy)

for iter in range(Net.args.epoch):
    test_temp = 0
    for exp_time in range(num_experiments):
        test_temp += exp_test_acc[exp_time][iter]
    final_test_accuracy.append(test_temp / num_experiments)

print(final_test_accuracy)
print('best test acc:', best_test)
df = pd.DataFrame(
    {'epoch': range(len(train_accuracy)), 'train_accuracy': train_accuracy,
     'test_accuracy': final_test_accuracy}, index=range(len(train_accuracy)))
df.to_csv(save_path + '_result.csv')