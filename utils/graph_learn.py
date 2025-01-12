import random
import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from utils import hyperedge as h


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default=0.001, help='learning rate')
parser.add_argument('--in_ch', type=int, default=187, help='feature dim')
parser.add_argument('--derate_step', type=int, default=10, help='step_size')
parser.add_argument('--gamma', type=float, default=0.9, help='scheduler shrinking rate')
parser.add_argument('--num_features', type=int, default=90, help='num_features')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
parser.add_argument('--num_epochs', type=int, default=200 , help='starting epoch')
parser.add_argument('--drop_out', type=int, default=0.6, help='drop_out')
parser.add_argument('--seed', type=int, default=3407, help='seed')
parser.add_argument('--top_k', type=int, default=7, help='number of hyperedge nodes')
opt = parser.parse_args()

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)


class Graph_Learn(nn.Module):
    def __init__(self, dropout, alpha, num_features, in_ch):#, input_num_features,output_num_features):
        super(Graph_Learn, self).__init__()
        self.dropout = dropout
        self.W = nn.Parameter(torch.empty(size=(in_ch, num_features))) # time_point num_features
        self.num_features = num_features
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.relu = nn.LeakyReLU(dropout)
        self.alpha = alpha
        self.a = nn.Parameter(torch.empty(size=(2 * num_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, ROISignals, plv):
        S = self.compute_similarity_matrix(ROISignals, plv)
        return S

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.num_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.num_features:, :])
        e = Wh1 + Wh2.T
        return self.relu(e)

    def compute_similarity_matrix(self, ROISignals, plv):
        Wh = torch.mm(ROISignals, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        e = F.softmax(e, dim=1)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(plv > 0.1, e, zero_vec)
        one = torch.ones(self.num_features,self.num_features)
        attention_new = one + attention
        atPLV = attention_new * plv
        return atPLV

    def F_norm_loss(self, S):
        return self.alpha * torch.sum(S ** 2)

    def diff_loss(self, S):
        diff = torch.abs(S - S.mean(dim=0, keepdim=True))
        return torch.mean(torch.sum(diff ** 2 * S, dim=(0, 1)))

def train_Graph(ROISignals, plv, num_features, dropout, alpha, num_epochs, lr):
    early_stopping_rounds = 2
    min_loss = 9999
    early_stopping_counter = 0
    model = Graph_Learn(dropout, alpha, num_features)
    ROISignals = ROISignals.T
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=5e-4)
    schedular = lr_scheduler.StepLR(optimizer, step_size=opt.derate_step, gamma=opt.gamma, last_epoch=-1);

    min_S = model(ROISignals, plv)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        S = model(ROISignals, plv)
        loss = model.F_norm_loss(S) + model.diff_loss(S)
        loss.backward()
        optimizer.step()
        schedular.step()

        if loss < min_loss:
            min_S = S
            min_loss = loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_rounds:
            break
    return min_S

def G(ROISigns,plv):
    return h.generate_G_from_H(h.hyperedge_concat(h.construct_H_with_diff(
        train_Graph(ROISigns, plv, opt.num_features, opt.drop_out, opt.alpha, opt.num_epochs, opt.lr), opt.top_k)))
