import torch.nn as nn

from model.layers.residual_block import ResidualBlock


class TCNBlock(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=5):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(ResidualBlock(input_dim if i == 0 else num_channels[i - 1], out_channels, kernel_size, dilation))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
