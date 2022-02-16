# -*- coding: utf-8 -*-
import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from sequence_models.config import Config
from sequence_models.model import TextCNN
from sequence_models.data import TextDataset
import argparse
import time
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer


def get_model():
    config = Config(sentence_max_size=100,
                batch_size=32,
                word_num=11000,
                label_num=13,
                hidden_size=100,
                learning_rate=0.005,
                epoch=2,
                out_channel=2)

    model = TextCNN(config)
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    checkpoint =  torch.load('sequence_models/checkpoints/random_init_non_fixed_hidden_100.ckpt')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
   



