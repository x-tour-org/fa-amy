import torch
import torch.nn.functional as F
import torch.nn as nn


class LIA1D(nn.Module):
    def __init__(self, channels, f=64):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, f, kernel_size=1)
        self.softpool = nn.AvgPool1d(kernel_size=7, stride=3, padding=0)
        self.conv2 = nn.Conv1d(f, f, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(f, channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.gate = nn.Sequential(nn.Sigmoid())
        kernel = torch.ones(3) / 3.0
        self.register_buffer('smooth_kernel', kernel.view(1, 1, 3))

    def forward(self, x):
        x = x.transpose(1, 2)
        g = self.gate(x[:, :1])
        w = self.sigmoid(self.conv3(self.conv2(self.softpool(self.conv1(x)))))
        w = F.interpolate(w, size=x.size(2), mode='nearest')
        w = F.conv1d(w, self.smooth_kernel.repeat(w.size(1), 1, 1), padding=1, groups=w.size(1))
        out = x * w * g
        return out.transpose(1, 2)
