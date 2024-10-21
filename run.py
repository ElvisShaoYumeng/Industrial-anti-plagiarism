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
from utils.train_utils import accuracy, average_weights
from utils.init_utils import seed_torch
from utils.maml_model import Net4CNN, Net4CNN_add, Net4CNN_PU, BP
import learn2learn as l2l
import optuna
import plotly

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
argparser.add_argument('--epoch', type=int, help=' ', default=1000)
argparser.add_argument('--train_meta_batch_size', type=int, help=' ', default=3)  #训练用户数量
argparser.add_argument('--valid_meta_batch_size', type=int, help=' ', default=3)
argparser.add_argument('--test_meta_batch_size' , type=int, help=' ', default=1)  #测试用户数量
argparser.add_argument('--local_epoch' , type=int, help=' ', default=1)
#argparser.add_argument('--local_batch_size' , type=int, help=' ', default=512)

args = argparser.parse_args(args = [])


#这个运行一次

def condition_map(filename):
    if '_0.mat' in filename:
        return 0
    elif '_1.mat' in filename:
        return 1
    elif '_2.mat' in filename:
        return 2
    elif '_3.mat' in filename:
        return 3
    else:
        pass


working_dir = Path('.')
DE12_path = Path("./Data/CWRU/raw/12k_DE")
NO_path = Path("./Data/CWRU/raw/Normal")

class_num = 10
n_each_class = 100
signal_length = 1024

df_NO = get_df_all(NO_path, segment_length=signal_length, normalize=True)
df_DE12 = get_df_all(DE12_path, segment_length=signal_length, normalize=True)

df_eachclass_dict = {k: df_DE12[df_DE12['label'] == k] for k in range(1, 10)}  # 每类样本的字典
df_eachclass_dict[0] = df_NO[df_NO['label'] == 0]

for v in df_eachclass_dict:  # 加入工况信息
    df_eachclass_dict[v]['condition'] = df_eachclass_dict[v]['filename'].apply(condition_map)


def gene_data(df_eachclass_dict, condition, class_num, n_each_class):  # 为某种工况产生数据
    # 该工况下的每类样本
    df_eachclass_on_condition_dict = {}
    for k, v in df_eachclass_dict.items():
        df_eachclass_on_condition_dict[k] = v[v['condition'] == condition]
        df_eachclass_on_condition_dict[k].index = range(len(df_eachclass_on_condition_dict[k]))
        # print(len(df_eachclass_on_condition_dict[k]))
    x = np.zeros((class_num, n_each_class, signal_length));
    y = np.zeros((class_num, n_each_class))
    for class_ in range(class_num):
        x[class_] = np.array(df_eachclass_on_condition_dict[class_].iloc[0:n_each_class, 2:-1])
        y[class_] = np.array(df_eachclass_on_condition_dict[class_].iloc[0:n_each_class, 0])
    return x, y


x_c0, y_c0 = gene_data(df_eachclass_dict, 0, class_num, n_each_class);
print(x_c0.shape, y_c0.shape);
np.save('./Data/CWRU/generated/x_c0.npy', x_c0);
np.save('./Data/CWRU/generated/y_c0.npy', y_c0)
x_c1, y_c1 = gene_data(df_eachclass_dict, 1, class_num, n_each_class);
print(x_c1.shape, y_c1.shape);
np.save('./Data/CWRU/generated/x_c1.npy', x_c1);
np.save('./Data/CWRU/generated/y_c1.npy', y_c1)
x_c2, y_c2 = gene_data(df_eachclass_dict, 2, class_num, n_each_class);
print(x_c2.shape, y_c2.shape);
np.save('./Data/CWRU/generated/x_c2.npy', x_c2);
np.save('./Data/CWRU/generated/y_c2.npy', y_c2)
x_c3, y_c3 = gene_data(df_eachclass_dict, 3, class_num, n_each_class);
print(x_c3.shape, y_c3.shape);
np.save('./Data/CWRU/generated/x_c3.npy', x_c3);
np.save('./Data/CWRU/generated/y_c3.npy', y_c3)


class Dataset_CWRU(data.Dataset):
    def __init__(self, args, condition, mode):
        super().__init__()
        self.sample_len = 1024
        # self.task_mode = True if ways == 10 else False
        self.__getdata__(mode, condition)

    def __getdata__(self, mode, condition):

        # 工况 K0 下的 10 分类   [类别，每一类取多少个，样本数]

        x = np.load('./Data/CWRU/generated/x_c' + str(condition) + '.npy')
        y = np.load('./Data/CWRU/generated/y_c' + str(condition) + '.npy')
        np.random.seed(1)
        np.random.shuffle(x)
        np.random.shuffle(y)

        if mode == 'train':
            x = x[:, 0:int(0.8 * x.shape[1]), :]
            y = y[:, 0:int(0.8 * y.shape[1])]
        elif mode == 'valid':
            x = x[:, int(0.8 * x.shape[1]):, :]
            y = y[:, int(0.8 * y.shape[1]):]
        elif mode == 'test':
            pass

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


def objective(trial):
    batch_size = trial.suggest_int("batch_size", 64, 256, step=64)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])  ##
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)  ##

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
    train_loaders = {};
    valid_loaders = {};
    test_loaders = {}
    for client in range(Net.args.train_meta_batch_size):
        train_loaders[client] = torch.utils.data.DataLoader(Dataset_CWRU(Net.args, condition=client, mode='train'),
                                                            batch_size=batch_size, shuffle=True)
        valid_loaders[client] = torch.utils.data.DataLoader(Dataset_CWRU(Net.args, condition=client, mode='valid'),
                                                            batch_size=batch_size, shuffle=True)
    for client in range(Net.args.test_meta_batch_size):
        test_loaders[client] = torch.utils.data.DataLoader(Dataset_CWRU(Net.args, condition=3, mode='test'),
                                                           batch_size=batch_size, shuffle=True)
    # 模型
    model = Net.model
    w_init = copy.deepcopy(model.state_dict())

    train_accuracy = [];
    valid_accuracy = [];
    test_accuracy = []  # 保存每个epoch的准确率
    best_test = 0  # 用于保存最佳模型

    for epoch in range(Net.args.epoch):
        t0 = time.time()
        train_acc_in_all_client = 0;
        valid_acc_in_all_client = 0;
        test_acc_in_all_client = 0

        w_locals = []
        for client in range(Net.args.train_meta_batch_size):
            if epoch == 0:
                model.load_state_dict(w_init)
            else:
                model.load_state_dict(w_glob)
            model.train()
            model, train_acc = Net.train(model, train_loaders[client])
            train_acc_in_all_client += train_acc
            w_locals.append(copy.deepcopy(model.state_dict()))
        w_glob = average_weights(w_locals)

        for client in range(Net.args.valid_meta_batch_size):
            model.load_state_dict(w_glob)
            model.eval()
            valid_acc = Net.valid(model, valid_loaders[client])
            valid_acc_in_all_client += valid_acc

        for client in range(Net.args.test_meta_batch_size):
            model.load_state_dict(w_glob)
            model.eval()
            test_acc = Net.test(model, test_loaders[client])
            test_acc_in_all_client += test_acc

        if test_acc_in_all_client / Net.args.test_meta_batch_size > best_test:
            best_test = test_acc_in_all_client / Net.args.test_meta_batch_size
            torch.save(model.state_dict(), save_path)

        train_accuracy.append(train_acc_in_all_client / Net.args.train_meta_batch_size)
        valid_accuracy.append(valid_acc_in_all_client / Net.args.valid_meta_batch_size)
        test_accuracy.append(test_acc_in_all_client / Net.args.test_meta_batch_size)

        ACCURACY = test_acc_in_all_client / Net.args.test_meta_batch_size
        trial.report(ACCURACY, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            # 打印参数
        t1 = time.time()
        if epoch % 20 == 0:
            print(f'Time /epoch: {t1 - t0:.4f} s', end='       ');
            print('Epoch', epoch + 1, end='       ')
            print(f'Train Accuracy:', round(train_acc_in_all_client / Net.args.train_meta_batch_size, 4), end='   ')
            print(f'Valid Accuracy:', round(valid_acc_in_all_client / Net.args.valid_meta_batch_size, 4), end='   ')
            print(f'Test  Accuracy:', round(test_acc_in_all_client / Net.args.test_meta_batch_size, 4))
    print('best test acc:', best_test)
    df = pd.DataFrame(
        {'epoch': range(len(train_accuracy)), 'train_accuracy': train_accuracy, 'valid_accuracy': valid_accuracy,
         'test_accuracy': test_accuracy}, index=range(len(train_accuracy)))
    df.to_csv(save_path + '_result.csv')

    return ACCURACY


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

df = study.trials_dataframe().drop(['state', 'datetime_start', 'datetime_complete', 'duration', 'number'], axis=1)
print(df)

df.to_csv('./df_results/FedAvg_PU_1.csv')

