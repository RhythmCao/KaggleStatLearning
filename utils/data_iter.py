#coding=utf8
import os,sys
import numpy as np
import torch
from torch.utils.data import Dataset
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class ImageDataset(Dataset):

    def __init__(self, data, label=None):
        super(ImageDataset, self).__init__()
        if label is not None:
            self.data, self.label = torch.from_numpy(data), torch.from_numpy(label)
        else:
            self.data, self.label = torch.from_numpy(data), None
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.label[idx]) if self.label is not None else self.data[idx] # all Tensor type

def collate_fn_for_data(batch):
    if type(batch[0]) == tuple:
        feats, label = list(zip(*batch))
        return torch.tensor(torch.stack(feats, dim=0), dtype=torch.float), torch.tensor(torch.stack(label, dim=0), dtype=torch.long)
    else:
        return torch.tensor(torch.stack(feats, dim=0), dtype=torch.float)

