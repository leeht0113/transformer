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
        self.src = self.tokenizer(list(self.data['src']), padding='max_length', truncation=True, max_length = self.max_len, return_tensors='pt').input_ids
        self.tar = self.tokenizer(['<s>' + s for s in self.data['tar']], padding='max_length', truncation=True, max_length = self.max_len, return_tensors='pt').input_ids
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.src[idx], self.tar[idx]
    

def dataloader(tokenizer, max_len, batch_size, train_split, valid_split, test_split):
    tokenizer = tokenizer
    tokenizer.add_special_tokens({'bos_token':'<s>'})
    train = pd.read_csv('preprocess.csv')
    custom_ds = CustomDataset(train, tokenizer, max_len)
    train_ds, valid_ds, test_ds = torch.utils.data.random_split(custom_ds, [train_split, valid_split, test_split])
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    return train_dl, valid_dl, test_dl