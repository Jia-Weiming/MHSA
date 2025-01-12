from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from torch.nn.parameter import Parameter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid1, dropout):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid1)
        self.hfc1 = HGNN_fc(in_ch, n_class)
        self.classify = torch.nn.Sequential(nn.Linear(in_ch, in_ch),
                                            nn.Dropout(dropout),
                                            nn.Linear(in_ch,n_class))

    def forward(self, x, G):
        G = torch.from_numpy(G).float().to('cuda')
        x = torch.from_numpy(x).float().to('cuda')
        x = F.dropout(x.T, self.dropout)
        x = F.relu(self.hgc1(x, G))
        x = x.mean(dim = 1)
        return self.classify(x)


class HGNNWrapper(torch.nn.Module):
    def __init__(self, in_ch, n_class, n_hid1, dropout):
        super(HGNNWrapper, self).__init__()
        self.hgnn = HGNN(in_ch, n_class, n_hid1, dropout)

    def forward(self, batch_X, batch_adj):
        gh_batch = []
        for idx in range(batch_X.shape[0]):
            tmp = self.hgnn(batch_X[idx],batch_adj[idx])
            gh_batch.append(tmp)
        gh_batch = torch.stack(gh_batch)
        return gh_batch

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = Parameter(torch.zeros(size=(in_ft, out_ft)))
        nn.init.xavier_normal_(self.weight.data, gain=1.4)
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight) # 权重和初始特征x相乘得到中间特征 x
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

class HGNN_fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)

class HGNN_classifier(nn.Module):
    def __init__(self, n_hid, n_class):
        super(HGNN_classifier, self).__init__()
        self.fc1 = nn.Linear(n_hid, n_class)
    def forward(self, x):
        x = self.fc1(x)
        return x

def z_score_standardization(matrix):
    mean = np.mean(matrix)
    std = np.std(matrix)
    return (matrix - mean) / std
