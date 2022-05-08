from torch import nn
import torch.nn.functional as F
import torch


class SVMBurnupModel(nn.Module):
    def __init__(self, num_ipt):
        super(SVMBurnupModel, self).__init__()
        self.fc1 = nn.Linear(num_ipt, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1)
        self.bn2 = nn.BatchNorm1d(1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        x = x.flatten()
        return x


