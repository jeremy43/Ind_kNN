from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

__all__ = ['Naive']


class Alexnet(nn.Module):

    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(Alexnet, self).__init__()
        model = torchvision.models.alexnet(pretrained=True)
        self.feature = nn.Sequential(*list(model.features.children()))
        self.fc = nn.Linear(256 * 1 * 1, 384)
        self.fc2 = nn.Linear(384, num_classes)

    def forward(self, x):
        feature2 = self.feature(x)
        feature2 = feature2.view(-1, 256 * 1)
        feature_1 = self.fc(feature2)
        feature_out = feature_1
        feature_1 = F.relu(feature_1)
        x = self.fc2(feature_1)
        return feature2, x
        return feature_out, x


class Naive(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(Naive, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1_drop = nn.Dropout2d()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128 * 25, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
        x = F.relu(self.bn2(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)))
        x = x.view(-1, 128 * 25)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        # output = x
        output = F.log_softmax(x)
        # print('output', output)
        f = x.view(x.size(0), -1)

        if not self.training:
            # print('return feature')
            return f, output
        return output
