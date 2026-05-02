import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """Dataset class for loading pre-computed protein embeddings."""

    def __init__(self, data_path):
        # Expecting data shape: [N, Sequence_Length, Embedding_Dim]
        self.data = np.load(data_path)

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float)

    def __len__(self):
        return len(self.data)
