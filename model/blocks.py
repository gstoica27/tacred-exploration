import torch
import numpy as np
import os
from torch import nn


class ReLUBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(ReLUBlock, self).__init__()
        self.projection = nn.Linear(input_size, output_size, bias=False)
        self.activation = nn.ReLU()
        self.network = nn.Sequential(self.projection, self.activation)

    def forward(self, inputs):
        return self.network(inputs)


class TanHBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(TanHBlock, self).__init__()

        self.network = nn.Sequential(nn.Linear(input_size, output_size, bias=False),
                                     nn.Tanh())

    def forward(self, inputs):
        return self.network(inputs)


class SigmoidBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(SigmoidBlock, self).__init__()

        self.network = nn.Sequential(nn.Linear(input_size, output_size, bias=False),
                                     nn.Sigmoid())

    def forward(self, inputs):
        return self.network(inputs)


class LinearBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearBlock, self).__init__()

        self.network = nn.Sequential(nn.Linear(input_size, output_size, bias=False))

    def forward(self, inputs):
        return self.network(inputs)


class NASCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NASCell, self).__init__()
        network_input_size = input_size + hidden_size
        # combine inputs
        self.linear0_0 = LinearBlock(network_input_size, hidden_size)
        # three relus at second layer
        self.relu1_0 = ReLUBlock(hidden_size, hidden_size)
        self.relu1_1 = ReLUBlock(hidden_size, hidden_size)
        self.relu1_2 = ReLUBlock(hidden_size, hidden_size)
        # third layer functions
        self.tanh2_0 = TanHBlock(hidden_size, hidden_size)
        self.relu2_1 = ReLUBlock(hidden_size, hidden_size)
        self.tanh2_2 = TanHBlock(hidden_size, hidden_size)
        self.tanh2_3 = TanHBlock(hidden_size, hidden_size)
        # fourth layer functions
        self.tanh3_0 = TanHBlock(hidden_size, hidden_size)
        self.tanh3_1 = TanHBlock(hidden_size, hidden_size)
        # fifth layer functions
        self.linear4_0 = LinearBlock(hidden_size, hidden_size)
        self.sigmoid4_1 = SigmoidBlock(hidden_size, hidden_size)

        self.branch0 = nn.Sequential(
            self.relu1_0,
            self.tanh2_0,
            self.tanh3_0,
            self.linear4_0
        )
        self.branch1 = nn.Sequential(
            self.relu1_0,
            self.relu1_1
        )
        self.branch2 = nn.Sequential(
            self.relu1_0,
            self.tanh2_0,
            self.tanh3_1
        )
        self.branch3 = nn.Sequential(
            self.relu1_0,
            self.tanh2_0,
            self.tanh3_0,
            self.sigmoid4_1
        )
        self.branch4 = nn.Sequential(
            self.relu1_1
        )
        self.branch5 = nn.Sequential(
            self.relu1_2,
            self.tanh2_2
        )
        self.branch6 = nn.Sequential(
            self.relu1_2,
            self.tanh2_3
        )
        self.branches = nn.ModuleList(
            [
                self.branch0,
                self.branch1,
                self.branch2,
                self.branch3,
                self.branch4,
                self.branch5,
                self.branch6
             ]
        )

    def forward(self, inputs, hidden):
        concat_inps = torch.cat([inputs, hidden], dim=-1)
        mixed_inps = self.linear0_0(concat_inps)

        output = self.branch0(mixed_inps)
        output += self.branch0(mixed_inps)
        output += self.branch1(mixed_inps)
        output += self.branch2(mixed_inps)
        output += self.branch3(mixed_inps)
        output += self.branch4(mixed_inps)
        output += self.branch5(mixed_inps)
        output += self.branch6(mixed_inps)
        output /= 7.
        return output


class NASRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NASRNN, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = NASCell(input_size=input_size, hidden_size=hidden_size)

    def forward(self, inputs, hidden, masks):

        outputs = []
        length = inputs.size()[1]
        for t in range(length):
            x_t = inputs[:, t, :]
            hidden = self.rnn(inputs=x_t, hidden=hidden)
            outputs.append(hidden)

        outputs = torch.stack(outputs, 1)
        # enforce EoS padding
        outputs *= masks.view(-1, length, 1)
        return outputs


if __name__ == '__main__':
    x = torch.rand(2, 5, 4)
    h = torch.rand(2, 5)
    rnn = NASRNN(4, 5)
    y = rnn(x, h)
    print(y)


