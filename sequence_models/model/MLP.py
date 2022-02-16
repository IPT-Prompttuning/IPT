# -*- coding: utf-8 -*-
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .BasicModule import BasicModule

class MLP(BasicModule):

    def __init__(self, config):
        super(MLP,self).__init__()
        self.config = config
        self.hidden = nn.Linear(1024,config.hidden_size*1024)
        self.tanh = nn.Tanh()
        self.hidden2 = nn.Linear(config.hidden_size*1024,1024)
        self.out = nn.Linear(1024,config.label_num)

    def forward(self, x):
        dropout = nn.Dropout(p=0.5)
        x = x.mean(dim=1)
        x1 = dropout(F.relu(self.hidden(x)))
        x2 = dropout(F.relu(self.hidden2(x1)))
        hidden_emb = x1.view(x1.shape[0],-1,1024)
        y = self.out(x2)
        return y,hidden_emb
