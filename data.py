import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, Data


class CloudTensorDataset(Dataset):

    def __init__(self, path_data):
        self.data = torch.load(path_data)
        self.data.len = len(self.data)

    def __len__(self):
        return self.data.len

    def __getitem__(self, idx):
        return self.data.__getitem__(idx)


class GeneratedData(InMemoryDataset):

    def __init__(self, root):
        super().__init__(root, None, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['generated-post-processed-clouds.pt']

    def process(self):
        boundaries = np.load(f'output/boundary.npy', allow_pickle=True)
        interiors = np.load(f'output/interior.npy', allow_pickle=True)
        data_list = []
        for boundary, interior in zip(boundaries, interiors):
            boundary, interior = torch.Tensor(boundary), torch.Tensor(interior)
            data_list.append(Data(interior=interior, boundary=boundary))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    dataset = GeneratedData('dataset')
    print(dataset[0])
