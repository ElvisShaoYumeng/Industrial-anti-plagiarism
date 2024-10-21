import os
import numpy as np
import pandas as pd

# Pytorch
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from pathlib import Path
from utils.helper_JNU import data_preprocessing

class Dataset_JNU(torch.utils.data.Dataset):
    def __init__(self, args, condition, spt_or_qry, label_list, location):
        super().__init__()
        self.args = args
        self.n_e_c = 400    #最多488 
        if not os.path.exists('./Data/JNU/generated/x_c0.npy'):
            if not os.path.exists('./Data/JNU/generated'):
                os.makedirs('./Data/JNU/generated')
            filename, data = data_preprocessing(dataset_path='../Data/JNU',sample_number=self.n_e_c,
                                                window_size=args.signal_length,overlap=args.signal_length,
                                                normalization='Z-score Normalization',
                                                noise=0,snr=0, input_type='TD')
            label = np.zeros((12,self.n_e_c));label[0:3] = 0;label[3:6] = 1;label[6:9] = 2;label[9:12] = 3
            x_c0, y_c0 = data[[0,3,6,9]] , label[[0,3,6,9]] ;np.save('./Data/JNU/generated/x_c0.npy', x_c0);np.save('./Data/JNU/generated/y_c0.npy', y_c0)
            x_c1, y_c1 = data[[1,4,7,10]], label[[1,4,7,10]];np.save('./Data/JNU/generated/x_c1.npy', x_c1);np.save('./Data/JNU/generated/y_c1.npy', y_c1)
            x_c2, y_c2 = data[[2,5,8,11]], label[[2,5,8,11]];np.save('./Data/JNU/generated/x_c2.npy', x_c2);np.save('./Data/JNU/generated/y_c2.npy', y_c2)
        self.__getdata__(condition, spt_or_qry, label_list, location)
    
    
    def __getdata__(self, condition, spt_or_qry, label_list, location):
        # 工况 condition下的 5 分类   [类别，每一类取多少个，样本数]
        x = np.load('./Data/JNU/generated/x_c'+str(condition)+'.npy')                   #;print(x.shape)    #[ways, num_each_class, signal_length]
        y = np.load('./Data/JNU/generated/y_c'+str(condition)+'.npy')                   #;print(y.shape)    #[ways, num_each_class]
        if label_list == 'all':  label_list = range(self.args.ways)
        k_s = self.args.k_spt
        k_q = self.args.k_qry
        if   spt_or_qry == 0:  x = x[label_list, location*(k_s+k_q):location*(k_s+k_q)+k_s, :]                                ;y = y[label_list, location*(k_s+k_q):location*(k_s+k_q)+k_s] 
        elif spt_or_qry == 1:  x = x[label_list, location*(k_s+k_q)+k_s:location*(k_s+k_q)+k_s+k_q,:] ;y = y[label_list,location*(k_s+k_q)+k_s:location*(k_s+k_q)+k_s+k_q]  
        else:                  x = x[label_list, location*(k_s+k_q):location*(k_s+k_q)+k_s+k_q,:]                ;y = y[label_list,location*(k_s+k_q):location*(k_s+k_q)+k_s+k_q] 

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