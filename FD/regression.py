import sys

sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
import torch.nn.init as init


class RegNet(nn.Module):
    def __init__(self):
        super(RegNet, self).__init__()
        # Batch x Channel x Height x Width; 128- feature dimension
        self.conv1 = nn.Conv2d(1, 32, [128, 1], 1).apply(kaiming_init)
        self.conv2 = nn.Conv2d(32, 1, 1, 1).apply(kaiming_init)
        self.fc1 = nn.Linear(128, 32).apply(kaiming_init)
        self.fc2 = nn.Linear(128, 32).apply(kaiming_init)
        self.fc3 = nn.Linear(64 + 1, 32).apply(kaiming_init)
        self.fc4 = nn.Linear(32, 1).apply(kaiming_init)
        self.dropout1 = nn.Dropout2d(0.15)
        self.dropout2 = nn.Dropout2d(0.15)
        self.dropout3 = nn.Dropout2d(0.5)

    def forward(self, x, y, f):
        # x: cov; y: mean
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout1(x)

        y = self.fc1(y)
        y = self.dropout2(y)

        z = torch.cat([x, y, f], dim=1)  # mean, variance, and fid
        z = self.fc3(z)
        z = self.dropout3(z)
        z = self.fc4(z)

        output = z.view(-1)
        return output


class REG(data.Dataset):
    """
    Args:
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, path, data, label, fid, transform=None, target_transform=None):
        super(REG, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.path = path
        self.label_file = label
        self.fid = fid

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (mean, var, target) where target is index of the target class.
        """
        mean = np.load(self.path + '_' + self.data[index] + '_mean.npy')
        var = np.load(self.path + '_' + self.data[index] + '_variance.npy')

        target = self.label_file[index]
        fid = self.fid[index]
        fid = torch.as_tensor(fid, dtype=torch.float).view(1)

        mean = torch.as_tensor(mean, dtype=torch.float)
        var = torch.as_tensor(var, dtype=torch.float).view(1, 128, 128)

        target = torch.as_tensor(target, dtype=torch.float)
        return var, mean, target, fid

    def __len__(self):
        return len(self.data)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
