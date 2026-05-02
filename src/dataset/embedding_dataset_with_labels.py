import torch
from torch.utils.data import Dataset


class BioinformaticsDataset(Dataset):
    """Custom dataset for loading protein embeddings and labels."""

    def __init__(self, label, prot):
        self.lb = label
        self.df_prot = prot

    def __getitem__(self, index):
        prot = torch.tensor(self.df_prot[index], dtype=torch.float)
        label = torch.tensor(self.lb[index], dtype=torch.float)
        return prot, label

    def __len__(self):
        return len(self.df_prot)