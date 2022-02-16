from torch.utils import data
import os
import linecache
import json
import torch
from transformers import RobertaConfig, AutoModelForMaskedLM, AutoTokenizer
from transformers import RobertaModel

class TextDataset(data.Dataset):
    #Preconditioning on the data for pretrain, sentences longer than the specified sentence length would be truncated.
    #We append '1' to sentences shorter than the specified length. 

    def __init__(self, path, dataset_name, tokenizer):
        self.tokenizer = tokenizer
        self.data_path = path + '/' + dataset_name
        self.config = RobertaConfig()
        self.model = RobertaModel(self.config).from_pretrained("roberta-large")
        self.raw_embedding = self.model.get_input_embeddings()

    def __getitem__(self, index):
        data_iter = linecache.getline(self.data_path, index+1)
        start = 0
        data_dict = json.loads(data_iter)
        label = data_dict['label']
        data = data_dict['text']

        data = self.tokenizer.encode(data)
        if(len(data) > 200):
            data = data[:200]
        while(len(data) < 200):
            data.append(1)
        data = torch.tensor(data, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return data, label


    def __len__(self):
        return len(open(self.data_path,'rU', encoding='utf-8',errors='ignore').readlines())


