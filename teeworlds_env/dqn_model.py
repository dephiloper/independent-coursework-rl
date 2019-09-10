import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def get_parameter_stats(parameters):
    mean_sum = 0
    std_sum = 0
    num_parameters = 0
    for param in parameters:
        num_parameters += 1
        mean_sum += param.mean().item()
        std_sum += param.std().item()

    return mean_sum, std_sum / num_parameters


class Net(nn.Module):
    def __init__(self, input_shape, n_actions, linear_layer_class):
        super(Net, self).__init__()

        self.linear_layer_class = linear_layer_class

        # in_channel of the first conv layer is the number color's / color depth of the image and represents later
        # the amount of channels, which means output filter of the first conv needs to be input of the second and so on
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.layers = [
            linear_layer_class(conv_out_size, 512),
            linear_layer_class(512, n_actions)
        ]

        self.fc = nn.Sequential(
            self.layers[0],
            nn.ReLU(),
            self.layers[1]
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)

    def get_stats(self):
        return get_parameter_stats(self.parameters())

    def noisy_layers_sigma_snr(self):
        if self.linear_layer_class == nn.Linear:
            raise TypeError("No noise layer used, so there is no possibility to calculate sigma snr.")

        return [
            ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item() for
            layer in self.layers
        ]


class DuelingNet(nn.Module):
    def __init__(self, input_shape, n_actions, linear_layer_class=nn.Linear):
        super(DuelingNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.linear_layer_class = linear_layer_class

        self.layers = [
            # advantage layers
            linear_layer_class(conv_out_size, 512),
            linear_layer_class(512, n_actions),

            # value layers
            linear_layer_class(conv_out_size, 512),
            linear_layer_class(512, 1)
        ]

        self.fc_adv = nn.Sequential(
            self.layers[0],
            nn.ReLU(),
            self.layers[1]
        )
        self.fc_val = nn.Sequential(
            self.layers[2],
            nn.ReLU(),
            self.layers[3]
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256.0
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean()

    def get_stats(self):
        return get_parameter_stats(self.parameters())

    def noisy_layers_sigma_snr(self):
        if self.linear_layer_class == NoisyLinear:
            return [
                ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item() for
                layer in self.layers
            ]

        raise AssertionError('noisy_layers_sigma_snr() is only implemented for NoisyLayers.')


class NoisyLinear(nn.Linear):
    """
    > create a matrix for sigma (mu will be stored in matrix inherited from nn.Linear)
    > to make sigma trainable wrap it in a nn.Parameter
    > register_buffer creates a tensor in the network which won't be updated by backprop, but will handled by nn.Module
    > extra param and buffer is created for the bias of the layer
    > sigma 0.017 init value from paper
    """

    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        """
        initialisation of the layer according to the paper
        """
        std = np.math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight.data, bias)
