from torch import nn
import torch.nn.functional as F


class BurnupModel(nn.Module):
    def __init__(self, num_ipt):
        super(BurnupModel, self).__init__()
        self.fc1 = nn.Linear(num_ipt, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.fc5 = nn.Linear(1024, 10)
        self.bn5 = nn.BatchNorm1d(10)
        self.fc6 = nn.Linear(10, 1)
        self.bn6 = nn.BatchNorm1d(1)

    def forward(self, x):
        # x: (bs, 23)
        x = F.relu(self.bn1(self.fc1(x)))  # (bs, 23) --> (bs, 256)
        x = F.relu(self.bn2(self.fc2(x)))  # (bs, 256) --> (bs, 1024)
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        # x = self.bn6(F.dropout(self.fc6(x)))  # (64, 1)
        x = self.bn6(self.fc6(x))
        x = x.flatten()  # (64)  x=x.view(-1)
        return x


