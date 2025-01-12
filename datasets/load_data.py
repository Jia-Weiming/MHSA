import os
import numpy as np
import torch
import utils.graph_learn as hy


def load_data(folder_path):
    # 指定要读取的文件夹路径
    folder_path = folder_path
    # 获取文件夹下的所有文件
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    list = []
    # 遍历文件列表，逐个读取txt文件内容
    for filename in file_list:
        file_path = os.path.join(folder_path, filename)
        data = np.loadtxt(file_path)
        # 取前 90 个脑区
        data = data[:, :90]
        list.append(data.T)
    array = np.array(list)
    return torch.Tensor(array)

def load_label(folder_path):
    data = np.loadtxt(folder_path)
    data = np.array(data)
    return torch.Tensor(data)

def load_G(x):
    x = z_score_standardization(x.numpy())
    x = torch.tensor(x)
    save_path = "D:/workstation/Python Projects/test20240707/PLV/"
    file_name = f"{'plv_'}{int(x[0][0] * 100000)}{'.txt'}"
    file_path = os.path.join(save_path, file_name)
    plv = np.loadtxt(file_path, encoding='utf-8')
    plv = torch.Tensor(plv)
    G = hy.G(x.T, plv)
    return G


def z_score_standardization(matrix):
    mean = np.mean(matrix)
    std = np.std(matrix)
    return (matrix - mean) / std

def get_data(dataroot):

    data = load_data(os.path.join(dataroot,"signal/"))
    label = load_label(os.path.join(dataroot,"labels/label.txt"))
    shuffle_idx = np.random.permutation(len(label))
    label = label[shuffle_idx]
    data = data[shuffle_idx]
    dataset = []
    for i in range(len(data)):
        temp = []
        # z_score = z_score_standardization(data[i].numpy())
        # temp.append(torch.tensor(z_score))
        temp.append(data[i])
        temp.append(load_G(data[i]))
        temp.append(label[i])
        # print(temp)
        dataset.append(temp)
    return dataset