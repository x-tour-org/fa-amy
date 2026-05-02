import torch
from torch import nn

from model.layers.tcn_block import TCNBlock


class BiTCN(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=5, merge_mode='concat'):
        super().__init__()
        self.forward_tcn = TCNBlock(input_dim, num_channels, kernel_size)
        self.backward_tcn = TCNBlock(input_dim, num_channels, kernel_size)
        self.merge_mode = merge_mode

    def forward(self, x):
        x_rev = torch.flip(x, dims=[-1])
        out_fwd = self.forward_tcn(x)
        out_bwd = torch.flip(self.backward_tcn(x_rev), dims=[-1])
        return torch.cat([out_fwd, out_bwd], dim=1) if self.merge_mode == 'concat' else out_fwd + out_bwd