# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from .BasicModule import BasicModule

class LSTM(BasicModule):

    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config
        self.bidirectional_set = True
        self.bidirectional = 2 if self.bidirectional_set else 1
        self.num_layers=2
        #bidirectional LSTM
        self.lstm = nn.LSTM(1024,64,num_layers=self.num_layers,bidirectional=self.bidirectional_set)
        self.lstm_to_hidden = nn.Linear(64*2,config.hidden_size*1024)
        self.hidden2label = nn.Linear(config.hidden_size*1024,config.label_num)
        self.batch_size = config.batch_size

    def init_hidden(self):
        h0 = Variable(torch.randn(self.num_layers*self.bidirectional, self.batch_size, 64)).cuda()
        c0 = Variable(torch.randn(self.num_layers*self.bidirectional, self.batch_size, 64)).cuda()
        return (h0,c0)

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x)
        hidden_return  = self.lstm_to_hidden(lstm_out[0])
        label_out = self.hidden2label(hidden_return)
        hidden_return = hidden_return.reshape(hidden_return.shape[0],-1,1024)
        return label_out,hidden_return
        
