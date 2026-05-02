import torch
from torch import nn

from model.layers.bitcn import BiTCN
from model.layers.lia_1d import LIA1D
from model.layers.multihead_attention import MultiHeadSelfAttention


class FAAmyModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Input dim 1152 corresponds to ESMC-600m embeddings
        self.bitcn = BiTCN(input_dim=1152, num_channels=[512, 256, 64])
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 1)
        self.drop = nn.Dropout(0.3)
        self.att1 = MultiHeadSelfAttention(input_dim=128, num_heads=4)
        self.att2 = LIA1D(128, f=64)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.bitcn(x.permute(0, 2, 1)).permute(0, 2, 1)
        fused = 0.5 * self.att1(x) + 0.5 * self.att2(x)

        pooled = torch.mean(x + fused, dim=1)

        x = self.drop(self.relu(self.fc2(pooled)))
        x = self.drop(self.relu(self.fc3(x)))
        return self.fc4(x)
