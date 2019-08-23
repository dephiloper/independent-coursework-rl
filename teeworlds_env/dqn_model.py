import numpy as np
import torch
from torch import nn
from torch.nn import Parameter


class Net(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Net, self).__init__()

        # in_channel of the first conv layer is the number color's / color depth of the image and represents later
        # the amount of channels, which means output filter of the first conv needs to be input of the second and so on
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)

    def get_stats(self):
        mean_sum = 0
        std_sum = 0
        num_parameters = 0
        for param in self.parameters():
            num_parameters += 1
            mean_sum += param.mean().item()
            std_sum += param.std().item()

        return mean_sum, std_sum/num_parameters

    def print_stats(self):
        mean, std = self.get_stats()
        print('net stats:')
        print(f'mean: {mean}')
        print(f'std: {std}')
