import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import time
from torch.cuda.amp import autocast, GradScaler
import random
from sklearn.metrics import roc_auc_score, log_loss
import re

class DAIGTDataset(Dataset):
    def __init__(self, text_list, tokenizer, max_len):
        self.text_list=text_list
        self.tokenizer=tokenizer
        self.max_len=max_len
    def __len__(self):
        return len(self.text_list)
    def __getitem__(self, index):
        text = self.text_list[index]
        tokenized = self.tokenizer(text=text,
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.max_len,
                                    return_tensors='pt')
        return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze()

class DAIGTDataLoader:
    def __init__(self, train_texts, val_texts, test_texts, tokenizer, max_len, batch_size, num_workers):
        self.train_texts = train_texts
        self.val_texts = val_texts
        self.test_texts = test_texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
    def get_loaders(self):
        train_dataset = DAIGTDataset(self.train_texts, self.tokenizer, self.max_len)
        val_dataset = DAIGTDataset(self.val_texts, self.tokenizer, self.max_len)
        test_dataset = DAIGTDataset(self.test_texts, self.tokenizer, self.max_len)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return train_loader, val_loader, test_loader
    