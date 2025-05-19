import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class lncRNA_encoder(nn.Module):
    def __init__(self, num_in_lncRNA, num_hidden, dropout, act=torch.tanh):
        super(lncRNA_encoder, self).__init__()
        self.num_in_lncRNA = num_in_lncRNA
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act = act
        self.W1 = nn.Parameter(torch.zeros(size=(self.num_in_lncRNA, self.num_hidden), dtype=torch.float))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.b1 = nn.Parameter(torch.zeros(self.num_hidden, dtype=torch.float))

    def forward(self, H_T):
        z1 = self.act(torch.mm(H_T, self.W1) + self.b1)
        return z1

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_in_lncRNA) + ' -> ' + str(self.num_hidden)



class disease_encoder(nn.Module):
    def __init__(self, num_in_disease, num_hidden, dropout, act=torch.tanh):
        super(disease_encoder, self).__init__()
        self.num_in_disease = num_in_disease
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act = act
        self.W1 = nn.Parameter(torch.zeros(size=(self.num_in_disease, self.num_hidden), dtype=torch.float))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.b1 = nn.Parameter(torch.zeros(self.num_hidden, dtype=torch.float))

    def forward(self, H):
        z1 = self.act(H.mm(self.W1) + 2*self.b1)
        return z1

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_in_disease) + ' -> ' + str(self.num_hidden)


class decoder2(nn.Module):
    def __init__(self, dropout=0.0, act=torch.sigmoid):
        super(decoder2, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = act

    def forward(self, z_lncRNA, z_disease):
        z_disease_ = self.dropout(z_disease)
        z_lncRNA_ = self.dropout(z_lncRNA)

        z = self.act(z_lncRNA_.mm(z_disease_.t()))
        return z


