import torch
import torch.nn.functional as F
from torch import nn

class FedAvgCNNwithBN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,32,kernel_size=5,padding=0,stride=1,bias=True)
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.activation1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=5,padding=0,stride=1,bias=True))
        self.bn2 = nn.BatchNorm2d(64)
        self.activation2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation2(out)
        out = self.pool2(out)

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out



class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out

