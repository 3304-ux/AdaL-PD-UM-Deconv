import os
import scipy.io
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SeismicDataset(Dataset):
    def __init__(self, y, D):
        self.y = y
        self.D = D

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return (
            self.y[idx],
            self.D,
        )

def collate_fn(batch):
    y_batch = torch.stack([item[0] for item in batch]).t()
    D_batch = batch[0][1]
    D_batch = D_batch.to(device)
    return y_batch, D_batch
def load_data():
    train_path = 'train_path'  # Input data file name
    train_data = scipy.io.loadmat(train_path)
    train_inputs = train_data['xxx']

    D_path = 'D_path'   # Dictionary data file name
    D_data = scipy.io.loadmat(D_path)
    D = torch.from_numpy(D_data['xxx']).float()

    train_dataset = SeismicDataset(
        y=torch.tensor(train_inputs, dtype=torch.float32),
        D=D.clone().detach().float(),
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)