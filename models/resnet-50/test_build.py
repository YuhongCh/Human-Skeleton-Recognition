from model import ResNet
import torch
from torchsummary import summary
from blocks import BottleneckBlock

def check_model():
    resnet50 = ResNet(BottleneckBlock, [3, 4, 6, 3], 64)
    resnet50.to(torch.device("cpu"))
    summary(resnet50, (3, 224, 224))


def check_CUDA():
    CUDA = torch.cuda.is_available()
    if CUDA:
        print("CUDA is available!!!")
    else:
        print("CUDA is not available...")


def main():
    check_model()

if __name__ == "__main__":
    main()