import math
import numpy as np
import numpy as np
import os

#数据归一化
def Normal_signal(data,normal_type):
    '''
    :param data: the data before normalization ---- shape:(samples,windows size)---egg:(100,1024)
    :param normal_type: the method of normalization
    :return data_norm : the data after normalization ---- shape:(samples,windows size)---egg:(100,1024)
    '''
    num, len = data.shape[0], data.shape[1]
    data_norm = np.zeros((num,len))  #创建一个全为0矩阵存放归一化后的数据
    for i in range(num):
        if normal_type == 'Z-score Normalization':
            mean, var = data[i].mean(axis=0), data[i].var(axis=0)
            data_norm[i] =(data[i] - mean) / np.sqrt(var)
            data_norm[i] =data_norm[i].astype("float32")

        elif normal_type == 'Max-Min Normalization':
            maxvalue, minvalue = data[i].max(axis=0), data[i].min(axis=0)
            data_norm[i] = (data[i] - minvalue) / (maxvalue - minvalue)
            data_norm[i] = data_norm[i].astype("float32")

        elif normal_type == '-1 1 Normalization':
            maxvalue, minvalue = data[i].max(axis=0), data[i].min(axis=0)
            data_norm[i] = -1 + 2 * ((data[i] - minvalue) / (maxvalue - minvalue))
            data_norm[i] = data_norm[i].astype("float32")

        else:
            print('the normalization is not existed!!!')
            break

    return data_norm

#定义滑窗采样及其样本重叠数
def Slide_window_sampling(data, window_size,overlap):
    '''
    :param data: the data raw signals with length n
    :param window_size: the sampling length of each samples
    :param overlap: the data shift length of neibor two samples
    :return squence: the data after sampling
    '''

    count = 0  # 初始化计数器
    data_length = int(data.shape[0])  # 信号长度
    sample_num = math.floor((data_length - window_size) / overlap + 1)  # 该输入信号得到的样本个数
    squence = np.zeros((sample_num, window_size), dtype=np.float32)  # 初始化样本数组
    for i in range(sample_num):
        squence[count] = data[overlap * i: window_size + overlap * i].T  # 更新样本数组
        count += 1  # 更新计数器

    return squence

#添加不同信噪比噪声
def Add_noise(x, snr):
    '''
    :param x: the raw sinal after sliding windows sampling
    :param snr: the snr of noise
    :return noise_signal: the data which are added snr noise
    '''
    d = np.random.randn(len(x))  # generate random noise
    P_signal = np.sum(abs(x) ** 2)
    P_d = np.sum(abs(d) ** 2)
    P_noise = P_signal / 10 ** (snr / 10)
    noise = np.sqrt(P_noise / P_d) * d
    noise_signal = x.reshape(-1) + noise
    return noise_signal


#傅里叶变换
def FFT(x):
    '''
    :param x: time frequency signal
    :return y: frequency signal
    '''
    #y = np.empty((x.shape[0],int(x.shape[1] / 2)))  #单边频谱
    y = np.empty((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        y[i] = (np.abs(np.fft.fft(x[i])) / len(x[i]))   #傅里叶变换、取幅值、归一化
        #y[i] = (np.abs(np.fft.fft(x[i])) / len(x[i]))[range(int(x[i].shape[0] / 2))]  # 傅里叶变换、取幅值、归一化、取单边
    return y


def data_preprocessing(dataset_path,sample_number,window_size,overlap,normalization,noise,snr,
                         input_type):

    root = dataset_path

    health = ['n600_3_2.csv', 'n800_3_2.csv', 'n1000_3_2.csv']  # 600 800 1000转速下的正常信号
    inner = ['ib600_2.csv', 'ib800_2.csv', 'ib1000_2.csv']  # 600 800 1000转速下的内圈故障信号
    outer = ['ob600_2.csv', 'ob800_2.csv', 'ob1000_2.csv']  # 600 800 1000转速下的外圈故障信号
    ball = ['tb600_2.csv', 'tb800_2.csv', 'tb1000_2.csv']  # 600 800 1000转速下的滚动体故障信号

    file_name = []  # 存放三种转速下、四种故障状态的文件名，一共12种类型
    file_name.extend(health)
    file_name.extend(inner)
    file_name.extend(outer)
    file_name.extend(ball)

    data1 = [[], [], [], [], [], [], [], [], [], [], [], []]  # 创建一个长度为12的空列表存放12种故障数据(每一类数据不平衡)
    for num, each_name in enumerate(file_name):
        dir = os.path.join(root, each_name)
        with open(dir, "r", encoding='gb18030', errors='ignore') as f:
            for line in f:
                line = float(line.strip('\n'))  # 删除每一行后的换行符号，并将字符型转化为数字
                data1[num].append(line)  # 将取出来的数据逐个存放到相应的列表中

    data = [[], [], [], [], [], [], [], [], [], [], [], []]  # 创建一个长度为12的空列表存放12种故障数据（每一类数据平衡）shape：(12,500500)
    for data1_i in range(len(data1)):
        data[data1_i].append(data1[data1_i][:500500])  # 将所有类型数据总长度截取为500500

    data = np.array(data).squeeze(axis=1)  # shape：(12,500500)

    # 添加噪声
    if noise == 1 or noise == 'y':
        noise_data = np.zeros((data.shape[0], data.shape[1]))
        for data_i in range(data.shape[0]):
            noise_data[data_i] = Add_noise(data[data_i], snr)
    else:
        noise_data = data

    # 滑窗采样
    sample_data = np.zeros((noise_data.shape[0], noise_data.shape[1] // window_size, window_size))
    for noise_data_i in range(noise_data.shape[0]):
        sample_data[noise_data_i] = Slide_window_sampling(noise_data[noise_data_i], window_size=window_size,
                                                          overlap=overlap)

    sample_data = sample_data[:, :sample_number, :]
    # 归一化
    if normalization != 'unnormalization':
        norm_data = np.zeros((sample_data.shape[0], sample_data.shape[1], sample_data.shape[2]))
        for sample_data_i in range(sample_data.shape[0]):
            norm_data[sample_data_i] = Normal_signal(sample_data[sample_data_i], normalization)
    else:
        norm_data = sample_data

    if input_type == 'TD':  #时域信号
        data = norm_data
    elif input_type == 'FD':  #频域信号
        data = np.zeros((norm_data.shape[0],norm_data.shape[1],norm_data.shape[2]))
        for label_index in range(norm_data.shape[0]):
            fft_data = FFT(norm_data[label_index,:,:])
            data[label_index,:,:] = fft_data
    return file_name, data




filename, data = data_preprocessing(dataset_path='./Data',
                                    sample_number=100,  #每类选取多少样本
                                    window_size=1024,
                                    overlap=1024,
                                    normalization='Z-score Normalization',
                                    noise=0,snr=0, input_type='TD')
label = np.zeros((12,100));label[0:3] = 0;label[3:6] = 1;label[6:9] = 2;label[9:12] = 3
x_c0, y_c0 = data[[0,3,6,9]] , label[[0,3,6,9]] ;np.save('x_c0.npy', x_c0);np.save('y_c0.npy', y_c0)    #工况0
x_c1, y_c1 = data[[1,4,7,10]], label[[1,4,7,10]];np.save('x_c1.npy', x_c1);np.save('y_c1.npy', y_c1)    #工况1
x_c2, y_c2 = data[[2,5,8,11]], label[[2,5,8,11]];np.save('x_c2.npy', x_c2);np.save('y_c2.npy', y_c2)    #工况2