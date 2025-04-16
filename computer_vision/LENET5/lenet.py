import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, FashionMNIST  # Gray Images Dataset


class LENET5(nn.Module):
    def __init__(self) -> None:
        """
        LENET5 has 5 layers with 2 CONV layers and 3 FC Layers.
        Only takes gray images i.e single channel unlike 3 channels for RGB Images.

        Input : 32X32
        Layers: 
        - CONV                                    - 6 Feature Maps @ 28x28 
        - Subsampling / Downsampling / MaxPooling - 6 @ 14x14
        - CONV                                    - 16 Feature Maps @ 10x10
        - Subsampling / Downsampling / MaxPooling - 16 @ 5x5
        - FC1                                     - 120
        - FC2                                     - 84
        - FC3                                     - 10
        """
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(),
            nn.MaxPool1d(),
        )
        self.layer2 = nn.Sequential(
             nn.Conv2d(),
            nn.MaxPool1d(),
        )
        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()
        self.fc2 = nn.Linear()


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)