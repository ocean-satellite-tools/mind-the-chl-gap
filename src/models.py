import torch
from torch import nn

import deepxde as dde


class ChlorophyllDeepONet(dde.nn.pytorch.deeponet.DeepONet):
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, activation):
        super().__init__(
            layer_sizes_branch, layer_sizes_trunk, activation, "Glorot normal"
        )

        self.branch_net = dde.nn.pytorch.fnn.FNN(
            layer_sizes_branch, activation, "Glorot normal"
        )
        self.trunk_net = dde.nn.pytorch.fnn.FNN(
            layer_sizes_trunk, activation, "Glorot normal"
        )

    def forward(self, inputs):
        x_func = self.branch_net(inputs)
        x_loc = self.trunk_net(inputs)
        if self._output_transform is not None:
            return self._output_transform(self.merge_branch_trunk(x_func, x_loc, -1))
        return self.merge_branch_trunk(x_func, x_loc, -1)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        x3 = self.decoder(x2)
        return x3
