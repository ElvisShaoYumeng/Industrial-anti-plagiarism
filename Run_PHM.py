import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils import data
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import argparse
import time
from datetime import datetime
import copy

import numpy as np
import random
from scipy.signal import resample
from tqdm import tqdm
from utils.helper_CWRU import get_df_all, download

#网络
from utils.train_utils import accuracy, average_weights, average_compt_weights
from utils.init_utils import seed_torch
from utils.maml_model import Net4CNN, Net4CNN_add, Net4CNN_PU, BP
#import learn2learn as l2l
import optuna
import plotly
from Noise_add import noise_add1

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq


class Reshape(object):
    def __call__(self, seq):
        #print(seq.shape)
        return seq.transpose()


class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)


class AddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)


class RandomAddGaussian(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            return seq + np.random.normal(loc=0, scale=self.sigma, size=seq.shape)



class RandomScale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            scale_factor = np.random.normal(loc=1, scale=self.sigma, size=(seq.shape[0], 1))
            scale_matrix = np.matmul(scale_factor, np.ones((1, seq.shape[1])))
            return seq*scale_matrix


class RandomStretch(object):
    def __init__(self, sigma=0.3):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            seq_aug = np.zeros(seq.shape)
            len = seq.shape[1]
            length = int(len * (1 + (random.random()-0.5)*self.sigma))
            for i in range(seq.shape[0]):
                y = resample(seq[i, :], length)
                if length < len:
                    if random.random() < 0.5:
                        seq_aug[i, :length] = y
                    else:
                        seq_aug[i, len-length:] = y
                else:
                    if random.random() < 0.5:
                        seq_aug[i, :] = y[:len]
                    else:
                        seq_aug[i, :] = y[length-len:]
            return seq_aug


class RandomCrop(object):
    def __init__(self, crop_len=20):
        self.crop_len = crop_len

    def __call__(self, seq):
        if np.random.randint(2):
            return seq
        else:
            max_index = seq.shape[1] - self.crop_len
            random_index = np.random.randint(max_index)
            seq[:, random_index:random_index+self.crop_len] = 0
            return seq

class Normalize(object):
    def __init__(self, type = "0-1"): # "0-1","-1-1","mean-std"
        self.type = type

    def __call__(self, seq):
        if  self.type == "0-1":
            seq =(seq-seq.min())/(seq.max()-seq.min())
        elif  self.type == "-1-1":
            seq = 2*(seq-seq.min())/(seq.max()-seq.min()) + -1
        elif self.type == "mean-std" :
            seq = (seq-seq.mean())/seq.std()
        else:
            raise NameError('This normalization is not included!')

        return seq

class dataset(Dataset):

    def __init__(self, list_data, test=False, transform=None):
        self.test = test
        if self.test:
            self.seq_data = list_data['data'].tolist()
        else:
            self.seq_data = list_data['data'].tolist()
            self.labels = list_data['label'].tolist()
        if transform is None:
            self.transforms = Compose([
                Reshape()
            ])
        else:
            self.transforms = transform


    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        if self.test:
            seq = self.seq_data[item]
            seq = self.transforms(seq)
            return seq, item
        else:
            seq = self.seq_data[item]
            label = self.labels[item]
            seq = self.transforms(seq)
            return seq, label



signal_size = 1024

# Case One
#Case1 = ['helical 1',"helical 2","helical 3", "helical 4"]
Case1 = ['spur '+str(i) for i in range(1,5)]
label1 = [i for i in range(4)]

#working condition
WC = {0:"30hz"+"_"+"High"+"_1",1:"35hz"+"_"+"High"+"_1",2:"40hz"+"_"+"High"+"_1",3:"45hz"+"_"+"High"+"_1"}

#generate Training Dataset and Testing Dataset
def get_files(root, condition, label, nec):
    data = []
    lab = []
    state1 = WC[condition]  # WC[0] can be changed to different working states
    path1 = os.path.join(root,Case1[label],Case1[label]+"_"+state1+".txt")
    data1, lab1 = data_load(path1,label=label, nec=nec)
    data += data1
    lab  += lab1
    return [data,lab]

def data_load(filename,label, nec):
    fl = np.loadtxt(filename,usecols=0)
    fl = fl.reshape(-1,1)
    data=[] 
    lab=[]
    start,end=0,signal_size
    n = 0
    while end<=fl.shape[0] and n<nec:
        data.append(fl[start:end])
        lab.append(label)
        start +=signal_size
        end +=signal_size
        n+=1
    return data, lab

#--------------------------------------------------------------------------------------------------------------------
class Dataset_PHM(object):
    #num_classes = 6     #Case 1 have 6 labels; Case 2 have 9 lables
    inputchannel = 1

    def __init__(self, args, condition, mode, location):
        super().__init__()
        self.sample_len = 1024
        self.args = args
        self.data_dir = './Data/PHM/raw'
        self.n_e_c = 100  ### 每一类数据量
        self.condition = condition
        self.data_transforms = Compose([Reshape(),Normalize("0-1"),Retype(),])
        if not os.path.exists('./Data/PHM/generated/x_c0.npy'):
            if not os.path.exists('./Data/PHM/generated'):
                os.makedirs('./Data/PHM/generated')
    
            x_c0, y_c0 = self.gene_data_of_given_condition( 0);print(x_c0.shape, y_c0.shape)
            np.save('./Data/PHM/generated/x_c0.npy', x_c0);np.save('./Data/PHM/generated/y_c0.npy', y_c0)
            x_c1, y_c1 = self.gene_data_of_given_condition( 1);print(x_c1.shape, y_c1.shape)
            np.save('./Data/PHM/generated/x_c1.npy', x_c1);np.save('./Data/PHM/generated/y_c1.npy', y_c1)
            x_c2, y_c2 = self.gene_data_of_given_condition( 2);print(x_c2.shape, y_c2.shape)
            np.save('./Data/PHM/generated/x_c2.npy', x_c2);np.save('./Data/PHM/generated/y_c2.npy', y_c2)
            x_c3, y_c3 = self.gene_data_of_given_condition( 3);print(x_c3.shape, y_c3.shape)
            np.save('./Data/PHM/generated/x_c3.npy', x_c3);np.save('./Data/PHM/generated/y_c3.npy', y_c3)
        
        self.__getdata__(condition, mode, location)
        
    
    def gene_data_of_given_condition(self, condition):                     #为某种工况产生数据并保存
        #该工况下的每类样本

        x = np.zeros((self.args.ways, self.n_e_c, self.args.signal_length,1));y = np.zeros((self.args.ways, self.n_e_c))
        for class_ in range(self.args.ways):
            list_data = get_files(self.data_dir, condition, class_, self.n_e_c)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            data = dataset(list_data=data_pd, transform=self.data_transforms)   
            x[class_] = np.array(data.seq_data)
            y[class_] = np.array(data.labels)
        return x, y

    def __getdata__(self, condition, mode, location):
        
        x = np.load('./Data/PHM/generated/x_c'+str(condition)+'.npy')
        y = np.load('./Data/PHM/generated/y_c'+str(condition)+'.npy')

        if mode == 'train':
            # 20&18
            x = x[:, 0 + 20 * location:16 + 20 * location, :]
            y = y[:, 0 + 20 * location:16 + 20 * location]
            # 16
            # x = x[:, 0 + 25 * location:20 + 25 * location, :]
            # y = y[:, 0 + 25 * location:20 + 25 * location]
            # 12
            # x = x[:, 0 + 33 * location:26 + 33 * location, :]
            # y = y[:, 0 + 33 * location:26 + 33 * location]
        elif mode == 'test':
            # 20&18
            x = x[:, 16 + 20 * location:20 + 20 * location, :]
            y = y[:, 16 + 20 * location:20 + 20 * location]
            # 16
            # x = x[:, 20 + 25*location:25 + 25*location, :]
            # y = y[:, 20 + 25*location:25 + 25*location]
            # 12
            # x = x[:, 26 + 33 * location:33 + 33 * location, :]
            # y = y[:, 26 + 33 * location:33 + 33 * location]

        self.x = np.array(x, dtype=float)
        self.y = np.array(y, dtype=float)
        self.x = self.x.reshape([-1, 1, self.sample_len])  # x: (n_way*n, len, 1), y: (n_way*n)
        self.y = self.y.reshape(-1)
    def __getitem__(self, item):
        x = self.x[item]  # (NC, l)
        y = self.y[item]
        return x, y

    def __len__(self):
        return len(self.x)

argparser = argparse.ArgumentParser()
#数据
argparser.add_argument('--signal_length', type=int, help='signal_length', default=1024)
argparser.add_argument('--ways', type=int, help='n way', default=4)
#argparser.add_argument('--shots', type=int, help=' ', default=5)
#argparser.add_argument('--train_task_num', type=int, help=' ', default=10000)
#argparser.add_argument('--valid_task_num', type=int, help=' ', default=10000)
#argparser.add_argument('--test_task_num',  type=int, help=' ', default=10000)
#argparser.add_argument('--sample_time',  type=int, help=' ', default=100)

#其他
argparser.add_argument('--device', type=int, help='n way', default=torch.device('cpu'))  #torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

#超参数
argparser.add_argument('--epoch', type=int, help=' ', default=200)
argparser.add_argument('--train_num', type=int, help=' ', default=20)  #训练用户数量
argparser.add_argument('--valid_num', type=int, help=' ', default=3)
argparser.add_argument('--test_num' , type=int, help=' ', default=20)  #测试用户数量
argparser.add_argument('--local_epoch' , type=int, help=' ', default=3)
#argparser.add_argument('--local_batch_size' , type=int, help=' ', default=512)

args = argparser.parse_args(args = [])

args.gpu = -1  ###cpu -1 gpu 1
batch_size = 256
optimizer_name = "Adam"  ##
lr = 0.0001  ##
lazy_ratio = 0  ## 抄袭比例
num_lazy = int(lazy_ratio * args.train_num)
noise_scale = 0
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

path = os.path.join(os.getcwd(), 'Saved_models_PHM', 'FedAvg',
                    datetime.strftime(datetime.now(), '%m%d-%H%M%S'));
save_path = path
Net = FedAvg_learner(args)

# 数据
train_loaders = {}
test_loaders = {}
#20
conditions = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3]
locations = [0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]
#16
# conditions = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
# locations = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
# 12
# conditions = [0,0,0,1,1,1,2,2,2,3,3,3]
# locations = [0,1,2,0,1,2,0,1,2,0,1,2]
for client in range(Net.args.train_num):
    train_loaders[client] = torch.utils.data.DataLoader(Dataset_PHM(Net.args, condition=conditions[client], mode='train', location= locations[client]),
                                                        batch_size=batch_size, shuffle=True)

for client in range(Net.args.train_num):
    test_loaders[client] = torch.utils.data.DataLoader(Dataset_PHM(Net.args, condition=conditions[client], mode='test', location= locations[client]),
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