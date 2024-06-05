import torch
from torch.utils.data import Dataset


class CloudTensorDataset(Dataset):

    def __init__(self, path_data):
        self.data = torch.load(path_data)
        self.data.len = len(self.data)

    def __len__(self):
        return self.data.len

    def __getitem__(self, idx):
        return self.data.__getitem__(idx)
