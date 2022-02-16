# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from config import Config
from model import TextCNN
from data import TextDataset
import argparse
import time
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

def accuracy(y_true, logits):
    acc = (logits.argmax(1) == y_true).float().mean()
    return acc.item()

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epoch', type=int, default=35)
parser.add_argument('--out_channel', type=int, default=2)
parser.add_argument('--label_num', type=int, default=13)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--load', type=bool, default=False)
parser.add_argument('--hidden_size', type=int, default=100)
args = parser.parse_args()


torch.manual_seed(args.seed)

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

# Create the configuration
config = Config(sentence_max_size=100,
                batch_size=args.batch_size,
                word_num=11000,
                label_num=args.label_num,
                learning_rate=args.lr,
                hidden_size=args.hidden_size,
                epoch=args.epoch,
                out_channel=args.out_channel)

training_set = TextDataset(path='data/knowledge_cat', dataset_name='train.jsonl', tokenizer = tokenizer)
validation_set = TextDataset(path='data/knowledge_cat', dataset_name='dev.jsonl', tokenizer = tokenizer)

training_iter = data.DataLoader(dataset=training_set,
                                batch_size=config.batch_size,
                                num_workers=0)

validation_iter = data.DataLoader(dataset=validation_set,
                                batch_size=500,
                                num_workers=0)

model = TextCNN(config)
embeds = torch.FloatTensor(50265,1024).uniform_(-0.5,0.5)
embeds = nn.Parameter(embeds, requires_grad=True)

if torch.cuda.is_available():
    model.cuda()
    embeds = embeds.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config.lr)

if(args.load):
    print('load checkpoints')
    checkpoint = torch.load('checkpoints/random_init_non_fixed.ckpt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

best_val_acc = -1

count = 0
loss_sum = 0
# Train the model
for epoch in range(config.epoch):
    for data, label in training_iter:
        if config.cuda and torch.cuda.is_available():
            data = data.cuda()
            labels = label.cuda()

        data = torch.unsqueeze(data, 1)
        data = data.cuda()
        data = data.long()

        input_data= None
        for i in range(data.shape[0]):
            if input_data is None:
                input_data = torch.index_select(embeds,0,data[i][0])
            else:
                tmp_embeds = torch.index_select(embeds,0,data[i][0])
                input_data = torch.cat((input_data,tmp_embeds),0)
        input_data = input_data.reshape(data.shape[0],-1,input_data.shape[-1])
        input_data = torch.unsqueeze(input_data,1)
        model.train()
        model.zero_grad()
        out,hidden = model(input_data)
        label = label.cuda()
        loss = criterion(out, autograd.Variable(label.long()))
        
        accuracy_train = accuracy(label, out)

        loss_sum += loss.item()
        count += 1
        if count % 100 == 0:
            print('epoch:', epoch, end=' ')
            print('The loss is:%.5f' % (loss_sum/100))
            
            loss_sum = 0
            count = 0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if count % 20 == 0:
            model.eval()
            with torch.no_grad():
                for data, label in validation_iter:
                    data = data.cuda()
                    label = label.cuda()
                    data = torch.unsqueeze(data,1)
                    input_data= None
                    for i in range(data.shape[0]):
                        if input_data is None:
                            input_data = torch.index_select(embeds,0,data[i][0])
                        else:
                            tmp_embeds = torch.index_select(embeds,0,data[i][0])
                            input_data = torch.cat((input_data,tmp_embeds),0)
                    input_data = input_data.reshape(data.shape[0],-1,input_data.shape[-1])
                    input_data = torch.unsqueeze(input_data,1)
                    out,hidden = model(input_data)
                     
                    accuracy_val = accuracy(label,out)
                    if accuracy_val > best_val_acc:
                        print('better val checkpoint save')
                        print('val acc:', accuracy_val)
                        best_val_acc = accuracy_val
                        path = 'checkpoints/random_init_non_fixed_hidden_100.ckpt'
                        torch.save(embeds,"checkpoints/pretrain_emb_hidden_100.pt")
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss
        },path)
