# Code is wirtten with reference to https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

import torch
from torch import nn
from blocks import Block, BottleneckBlock

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()