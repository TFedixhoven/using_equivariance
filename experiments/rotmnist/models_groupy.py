'''
CODE RETRIEVED FROM https://github.com/Attila94/SepGroupPy/blob/main/experiments/rotmnist/models.py
'''

import torch.nn as nn
import torch.nn.functional as F

from sepgroupy.gconv.pooling import GroupCosetMaxPool, GroupMaxPool2d
from sepgroupy.gconv.splitgconv2d import P4ConvZ2, P4ConvP4

class Net(nn.Module):

    def __init__(self, planes, in_conv, hidden_conv, bn, mp, gmp,
                 num_classes=10, equivariance='P4'):
        super(Net, self).__init__()
        self.equivariance = equivariance

        self.conv1 = in_conv(3, planes, kernel_size=3, padding=1)
        self.conv2 = hidden_conv(planes, planes, kernel_size=3, padding=1)
        self.conv3 = hidden_conv(planes, planes, kernel_size=3, padding=1)
        self.conv4 = hidden_conv(planes, planes, kernel_size=3, padding=1)
        self.conv5 = hidden_conv(planes, planes, kernel_size=3, padding=1)
        self.conv6 = hidden_conv(planes, planes, kernel_size=3, padding=1)
        self.conv7 = hidden_conv(planes, planes, kernel_size=4)

        self.fc = nn.Linear(planes, num_classes)

        self.bn1 = bn(planes)
        self.bn2 = bn(planes)
        self.bn3 = bn(planes)
        self.bn4 = bn(planes)
        self.bn5 = bn(planes)
        self.bn6 = bn(planes)
        self.bn7 = bn(planes)

        self.mp = mp(2)
        self.gmp = gmp() if gmp is not None else None
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.mp(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))

        if self.equivariance == 'P4':
            s = x.shape
            x = x.view((s[0], s[1] * s[2], s[3], s[4]))

        x = self.avgpool(x)

        if self.gmp:
            x = x.view((s[0], s[1], s[2], 1, 1))
            x = self.gmp(x)

        x = x.squeeze()
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def SepCNN(planes, equivariance='P4', num_classes=10):

    assert equivariance in ['Z2', 'P4'], 'Must be one of Z2, P4'

    if equivariance == 'Z2':
        in_conv = nn.Conv2d
        hidden_conv = nn.Conv2d
        bn = nn.BatchNorm2d
        mp = nn.MaxPool2d
        gmp = None
    else:
        in_conv = P4ConvZ2
        hidden_conv = P4ConvP4
        bn = nn.BatchNorm3d
        mp = GroupMaxPool2d
        gmp = GroupCosetMaxPool

    return Net(planes, in_conv, hidden_conv, bn, mp, gmp, num_classes, equivariance=equivariance)