# Code is wirtten with reference to https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

from typing import Union
from torch import nn, Tensor
import torch
from blocks import Block, BottleneckBlock

class ResNet(nn.Module):
    """
    NOTE: The complete resnet-50 model should have include layer4 which contains most of parameters
          However, due to it's a personal project and it cost too much to train all these parameters, it is reduced
    """
    def __init__(self, block: Union[Block, BottleneckBlock],
                       layers: list[int], 
                       inplanes: int = 64,
                       outplanes: int = 2,
                       zero_init_residual: bool = False, 
                       groups: int = 1,
                       width_per_group: int = 64,
                       replace_stride_with_dilation: list[bool] = [False, False, False]):
        super().__init__()

        # initiate needed hyperparameters
        self.inplanes = inplanes
        self.dilation = 1
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(f"replace_stride_with_dilation should have length 3 but get {len(replace_stride_with_dilation)}")
        if len(layers) < 3:
            raise ValueError(f"layers parameter should at least have length of 3 but get {len(layers)}")

        # create layers 
        self.groups = groups
        self.width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layers(block, 64, layers[0])
        self.layer2 = self.make_layers(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self.make_layers(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        # self.layer4 = self.make_layers(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion_factor, outplanes)

        ##########################################################################################
        # there are some initialization weight setup, Complete copy paste from github link above #
        ##########################################################################################
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckBlock):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, Block):
                    nn.init.constant_(m.bn2.weight, 0)


    def make_layers(self, block: Union[Block, BottleneckBlock], inplanes: int, block_num: int, stride: int = 1, dilate: bool = False) -> nn.Sequential:
        if dilate:
            self.dilution *= stride
            stride = 1
        downsample = None
        if stride != 1 or self.inplanes != inplanes * block.expansion_factor:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, inplanes * block.expansion_factor, kernel_size=1, stride=stride),
                nn.BatchNorm2d(inplanes * block.expansion_factor)
            )

        layers = []
        layers.append(block(self.inplanes, inplanes, stride=stride, downsample=downsample))
        self.inplanes = inplanes * block.expansion_factor
        for _ in range(1, block_num):
            layers.append(block(self.inplanes, inplanes))

        return nn.Sequential(*layers)

    
    def forward(self, input: Tensor) -> Tensor:
        out = self.conv1(input)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
