# -*- coding: utf-8 -*-
import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from sequence_models.config import Config
from sequence_models.model.MLP import *
from sequence_models.data import TextDataset
import argparse
import time
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

#load pretrained model to IPT
def get_model():
    config = Config(sentence_max_size=100,
                batch_size=32,
                word_num=11000,
                label_num=13,
                hidden_size=100,
                learning_rate=0.0001,
                epoch=2,
                out_channel=2)

    model = MLP(config)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    checkpoint =  torch.load('sequence_models/checkpoints/mlp.ckpt')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
   


