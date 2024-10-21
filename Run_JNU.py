import os
import numpy as np
import pandas as pd

# Pytorch
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import argparse
import time
from datetime import datetime
import copy
import random
from scipy.signal import resample

from pathlib import Path
from helper_JNU import data_preprocessing
from Noise_add import noise_add1
from train_utils import accuracy, average_weights, average_compt_weights
from maml_model import Net4CNN, Net4CNN_add, Net4CNN_PU, BP

class Dataset_JNU(torch.utils.data.Dataset):
    def __init__(self, args, condition, mode, location):
        super().__init__()
        self.args = args
        self.n_e_c = 400    #最多488 
        if not os.path.exists('./Data/JNU/generated/x_c0.npy'):
            if not os.path.exists('./Data/JNU/generated'):
                os.makedirs('./Data/JNU/generated')
            filename, data = data_preprocessing(dataset_path='./Data/JNU',sample_number=self.n_e_c,
                                                window_size=args.signal_length,overlap=args.signal_length,
                                                normalization='Z-score Normalization',
                                                noise=0,snr=0, input_type='TD')
            label = np.zeros((12,self.n_e_c));label[0:3] = 0;label[3:6] = 1;label[6:9] = 2;label[9:12] = 3
            x_c0, y_c0 = data[[0,3,6,9]] , label[[0,3,6,9]] ;np.save('./Data/JNU/generated/x_c0.npy', x_c0);np.save('./Data/JNU/generated/y_c0.npy', y_c0)
            x_c1, y_c1 = data[[1,4,7,10]], label[[1,4,7,10]];np.save('./Data/JNU/generated/x_c1.npy', x_c1);np.save('./Data/JNU/generated/y_c1.npy', y_c1)
            x_c2, y_c2 = data[[2,5,8,11]], label[[2,5,8,11]];np.save('./Data/JNU/generated/x_c2.npy', x_c2);np.save('./Data/JNU/generated/y_c2.npy', y_c2)
        self.__getdata__(condition, mode, location)
    
    
    def __getdata__(self, condition, mode, location):
        # 工况 condition下的 5 分类   [类别，每一类取多少个，样本数]
        x = np.load('./Data/JNU/generated/x_c'+str(condition)+'.npy')                   #;print(x.shape)    #[ways, num_each_class, signal_length]
        y = np.load('./Data/JNU/generated/y_c'+str(condition)+'.npy')                   #;print(y.shape)    #[ways, num_each_class]

        if mode == 'train':
            #20
            # x = x[:, 0 + 55 * location:44 + 55 * location, :]
            # y = y[:, 0 + 55 * location:44 + 55 * location]
            #18
            x = x[:, 0 + 66 * location:53 + 66 * location, :]
            y = y[:, 0 + 66 * location:53 + 66 * location]
            #15
            # x = x[:, 0 + 80 * location:64 + 80 * location, :]
            # y = y[:, 0 + 80 * location:64 + 80 * location]
        elif mode == 'test':
            #20
            # x = x[:, 44 + 55 * location:55 + 55 * location, :]
            # y = y[:, 44 + 55 * location:55 + 55 * location]
            #18
            x = x[:, 53 + 66 * location:66 + 66 * location, :]
            y = y[:, 53 + 66 * location:66 + 66 * location]
            #15
            # x = x[:, 64 + 80 * location:80 + 80 * location, :]
            # y = y[:, 64 + 80 * location:80 + 80 * location]

        self.x = x.reshape([-1, 1, self.args.signal_length])                        #;print(self.x.shape)    #[ways*num_each_class(train / valid / test), 1, signal_length]
        self.y = y.reshape(-1)                                                      #;print(self.y.shape)    #[ways*num_each_class(train / valid / test)]
        self.x = torch.from_numpy(self.x)                                           #;print(self.x.shape)
        self.y = torch.from_numpy(self.y)                                           #;print(self.y.shape)
        
    def __getitem__(self, item):
        x = self.x[item]  
        y = self.y[item]
        return x, y 
    def __len__(self):
        return len(self.x)

argparser = argparse.ArgumentParser()
#数据
argparser.add_argument('--signal_length', type=int, help='signal_length', default=1024)
argparser.add_argument('--ways', type=int, help='n way', default=12)
#argparser.add_argument('--shots', type=int, help=' ', default=5)
#argparser.add_argument('--train_task_num', type=int, help=' ', default=10000)
#argparser.add_argument('--valid_task_num', type=int, help=' ', default=10000)
#argparser.add_argument('--test_task_num',  type=int, help=' ', default=10000)
#argparser.add_argument('--sample_time',  type=int, help=' ', default=100)

#其他
argparser.add_argument('--device', type=int, help='n way', default=torch.device('cpu'))  #torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

#超参数
argparser.add_argument('--epoch', type=int, help=' ', default=200)
argparser.add_argument('--train_num', type=int, help=' ', default=18)  #训练用户数量
argparser.add_argument('--valid_num', type=int, help=' ', default=3)
argparser.add_argument('--test_num' , type=int, help=' ', default=18)  #测试用户数量
argparser.add_argument('--local_epoch' , type=int, help=' ', default=3)
#argparser.add_argument('--local_batch_size' , type=int, help=' ', default=512)

args = argparser.parse_args(args = [])
args.gpu = -1  ###cpu -1 gpu 1
batch_size = 256
optimizer_name = "Adam"  ##
lr = 0.0001  ##
lazy_ratio = 0  ## 抄袭比例
num_lazy = int(lazy_ratio * args.train_num)
noise_scale = 0.000011
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

path = os.path.join(os.getcwd(), 'Saved_models_JNU', 'FedAvg',
                    datetime.strftime(datetime.now(), '%m%d-%H%M%S'));
save_path = path
Net = FedAvg_learner(args)

# 数据
train_loaders = {}

test_loaders = {}
#20
# conditions = [0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2]
# locations = [0,1,2,3,4,5,6,0,1,2,3,4,5,6,0,1,2,3,4,5]
#18
conditions = [0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2]
locations = [0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5]
#15
# conditions = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2]
# locations = [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]
for client in range(Net.args.train_num):
    train_loaders[client] = torch.utils.data.DataLoader(Dataset_JNU(Net.args, condition=conditions[client], mode='train', location= locations[client]),
                                                        batch_size=batch_size, shuffle=True)

for client in range(Net.args.train_num):
    test_loaders[client] = torch.utils.data.DataLoader(Dataset_JNU(Net.args, condition=conditions[client], mode='test', location= locations[client]),
                                                       batch_size=batch_size, shuffle=True)
# 模型
model = Net.model
w_init = copy.deepcopy(model.state_dict())

train_accuracy = []

test_accuracy = []  # 保存每个epoch的准确率
best_test = 0  # 用于保存最佳模型
exp_train_acc = []
exp_test_acc = []
final_test_accuracy = []

for exp_time in range(num_experiments):
    train_accuracy = []
    test_accuracy = []  # 保存每个epoch的准确率
    for epoch in range(Net.args.epoch):
        t0 = time.time()
        train_acc_in_all_client = 0;

        test_acc_in_all_client = 0

        w_locals = []
        for client in range(Net.args.train_num):
            if epoch == 0:
                model.load_state_dict(w_init)
            else:
                model.load_state_dict(w_glob)
            if client < (Net.args.train_num - num_lazy):
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
        rho = [1 / args.train_num] * args.train_num
        ### 抄袭检测
        # rho1 = [1 / (args.train_num - num_lazy)] * (args.train_num - num_lazy)
        # rho2 = [0] * num_lazy
        # rho = rho1 + rho2
        w_glob = average_compt_weights(w_locals,rho)

        ###  基于贡献度的聚合  ###
        ### 计算各用户贡献度
        # contri_list = []
        # for contri_idx in range(Net.args.train_num - rho.count(0)):
        #     temp_rho = copy.deepcopy(rho)
        #     temp_rho[contri_idx] = 0  ### 生成排除单个用户的全局模型
        #     for idx in range(len(temp_rho)):
        #         if temp_rho[idx] > 0:
        #             temp_rho[idx] = 1 / (Net.args.train_num - temp_rho.count(0))
        #     w_exg = average_compt_weights(w_locals,temp_rho)
        #     diff_list = []
        #     for client in range(Net.args.train_num): ###在其他用户端测试精度
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
        for client in range(Net.args.train_num):
            model.load_state_dict(w_glob)
            model.eval()
            test_acc = Net.test(model, test_loaders[client])
            test_acc_in_all_client += test_acc

        if test_acc_in_all_client / Net.args.train_num > best_test:
            best_test = test_acc_in_all_client / Net.args.train_num
            torch.save(model.state_dict(), save_path)

        train_accuracy.append(train_acc_in_all_client / Net.args.train_num)

        test_accuracy.append(test_acc_in_all_client / Net.args.train_num)

        ACCURACY = test_acc_in_all_client / Net.args.train_num



            # 打印参数
        t1 = time.time()
        if epoch % 20 == 0:
            print(f'Time /epoch: {t1 - t0:.4f} s', end='       ');
            print('Epoch', epoch + 1, end='       ')
            print(f'Train Accuracy:', round(train_acc_in_all_client / Net.args.train_num, 4), end='   ')

            print(f'Test  Accuracy:', round(test_acc_in_all_client / Net.args.train_num, 4))

    exp_train_acc.append(train_accuracy)
    exp_test_acc.append(test_accuracy)
    print('best test acc:', best_test)

for iter in range(Net.args.epoch):
    test_temp = 0
    for exp_time in range(num_experiments):
        test_temp += exp_test_acc[exp_time][iter]
    final_test_accuracy.append(test_temp / num_experiments)

print(exp_test_acc)
print(final_test_accuracy)

df = pd.DataFrame(
    {'epoch': range(len(train_accuracy)), 'train_accuracy': train_accuracy,
     'test_accuracy': final_test_accuracy}, index=range(len(train_accuracy)))
df.to_csv(save_path + '_result.csv')