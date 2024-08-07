import pandas as pd
import numpy as np
import re
import torch
from transformers import AutoTokenizer

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.src = self.tokenizer(list(self.data['src']), padding=True, truncation=True, max_length = self.max_len, return_tensors='pt').input_ids
        self.tar = self.tokenizer(['<s>' + s for s in self.data['tar']], padding=True, truncation=True, max_length = self.max_len, return_tensors='pt').input_ids
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.src[idx], self.tar[idx]
    

def dataloader(max_len):
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    train = pd.read_csv('preprocess.csv')
    custom_ds = CustomDataset(train, tokenizer, max_len)
    train_ds, test_ds = torch.utils.data.random_split(custom_ds, [0.9, 0.1])
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=True)
    return train_dl, test_dl