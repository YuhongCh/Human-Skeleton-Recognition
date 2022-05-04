# Code is wirtten with reference to https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

from torch import Tensor, nn

class Block(nn.Module):
    """
    This is just a basic block to contruct resnet.
    """
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int = 3, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, 
                               out_channels=out_channel,
                               kernel_size=kernel_size,
                               stride=stride)
        self.conv2 = nn.Conv2d(in_channels=out_channel, 
                               out_channels=out_channel,
                               kernel_size=kernel_size,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLu()
        self.downsample = downsample

    def forward(self, input: Tensor) -> Tensor:
        copy = Tensor.copy_(input)

        out = self.conv1(input)
        out = self.relu(self.bn1(out))
        out = self.conv2(input)
        out = self.bn2(out)

        if self.downsample is not None:
            copy = self.downsample(input)
        
        out = self.relu(out + copy)
        return out


class BottleneckBlock(nn.Module):
    """
    Similar to Blocks class above but use 1x1 conv layer to create bottleneck
    """
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int = 3, stride: int = 1, expansion_factor: int = 4, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, 
                               out_channels=out_channel,
                               kernel_size=1,
                               stride=stride)
        self.conv2 = nn.Conv2d(in_channels=out_channel, 
                               out_channels=out_channel,
                               kernel_size=kernel_size,
                               stride=stride)
        self.conv3 = nn.Conv2d(in_channels=out_channel, 
                               out_channels=out_channel * expansion_factor,
                               kernel_size=kernel_size,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.bn3 = nn.BatchNorm2d(out_channel * expansion_factor)
        self.relu = nn.ReLu()
        self.downsample = downsample

    def forward(self, input: Tensor) -> Tensor:
        copy = Tensor.copy_(input)

        out = self.conv1(input)
        out = self.relu(self.bn1(out))
        out = self.conv2(input)
        out = self.relu(self.bn2(out))
        out = self.conv3(input)
        out = self.bn3(out)

        if self.downsample is not None:
            copy = self.downsample(input)
        
        out = self.relu(out + copy)
        return out
