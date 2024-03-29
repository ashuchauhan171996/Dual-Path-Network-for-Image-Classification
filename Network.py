### YOUR CODE HERE
import torch
import torch.nn as nn

from torch.functional import Tensor

"""This script defines the network.
"""


class DualPathNetwork(nn.Module):
    """
    Dual Path Network (combination of Dense-Net and Res-Net) | Output from block is splited into two 
    with one used as res-net and other used as Dense-net
    
    """

    def __init__(self, configs):
        super(DualPathNetwork, self).__init__()
        self.configs = configs
        
        dense_depths = [16, 32, 24, 128]
        block_size = [3, 4, 20, 3]
        in_channel = [64, 128, 256, 512]
        out_channel = [128, 256, 512, 1024]

        # conversion of features from 3 to 64 --> start layer
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()

        # define the stack layers
        self.prev_channel = 64
        self.stack1 = self._layer(in_channel[0], out_channel[0], dense_depths[0], 1, block_size[0])
        self.stack2 = self._layer(in_channel[1], out_channel[1], dense_depths[1], 2, block_size[1])
        self.stack3 = self._layer(in_channel[2], out_channel[2], dense_depths[2], 2, block_size[2])
        self.stack4 = self._layer(in_channel[3], out_channel[3], dense_depths[3], 2, block_size[3])

        # output layer
        self.avg_pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(in_features=out_channel[3] + (block_size[3]+1)*dense_depths[3], out_features=10, bias=True)

    def _layer(self, in_channel, out_channel, dense_depth, stride, block_size):
        """
        Creates stack layers of Dual Path Network
        """
        stack_layers = []
        stride_list = [1] * block_size
        stride_list[0] = stride
        first_layer = True
        for i in range(block_size):
            if i != 0:
                first_layer = False
            stack_layers.append(Bottleneck(self.prev_channel, in_channel, out_channel, dense_depth, stride_list[i],
                                           first_layer))
            self.prev_channel = out_channel + (i+2)*dense_depth
        stack = nn.Sequential(*stack_layers)
        return stack

    def forward(self, inputs: Tensor):
       
        out = self.conv(inputs)
        out = self.bn(out)
        out = self.relu(out)

        out = self.stack1(out)
        out = self.stack2(out)
        out = self.stack3(out)
        out = self.stack4(out)
        out = self.avg_pool(out)
        
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Bottleneck(nn.Module):
    """
    Creates a bottleneck layer to be used in Dual Path Network
    """

    def __init__(self, prev_channel, in_channel, out_channel, dense_depth, stride, first_layer) -> None:
        super(Bottleneck, self).__init__()
        self.out_channel = out_channel
        self.conv1 = nn.Conv2d(in_channels=prev_channel, out_channels=in_channel, stride=1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=in_channel)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, stride=stride, kernel_size=3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=in_channel)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel + dense_depth, stride=1, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channel + dense_depth)
        self.relu3 = nn.ReLU()

        self.shortcut = nn.Sequential()
        if first_layer:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=prev_channel, out_channels=out_channel + dense_depth, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(num_features=out_channel + dense_depth) )
        self.relu4 = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
    
        # 1st conv layer
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu1(out)

        # 2nd conv layer
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # 3rd conv layer
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        inputs = self.shortcut(inputs)
        split = self.out_channel

        #spliting the output for dense path and residual path- size of split for residual path should be equal to original pipe size
        out = torch.cat([inputs[:, :split, :, :] + out[:, :split, :, :], inputs[:, split:, :, :], out[:, split:, :, :]],1)
        out = self.relu4(out)
        return out


### END CODE HERE
