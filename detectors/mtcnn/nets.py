import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np


PATH_PNET = './detectors/mtcnn/weights/pnet.npy'
PATH_RNET = './detectors/mtcnn/weights/rnet.npy'
PATH_ONET = './detectors/mtcnn/weights/onet.npy'


class Flatten(nn.Module):

    def __init__(self):

        super(Flatten, self).__init__()

    def forward(self, x):

        # without this pretrained model isn't working
        x = x.transpose(3, 2).contiguous()

        return x.view(x.size(0), -1)


class PNet(nn.Module):

    def __init__(self):

        super(PNet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 10, 3, 1)),
            ('prelu1', nn.PReLU(10)),
            ('pool1', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(10, 16, 3, 1)),
            ('prelu2', nn.PReLU(16)),

            ('conv3', nn.Conv2d(16, 32, 3, 1)),
            ('prelu3', nn.PReLU(32))
        ]))

        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

        weights = np.load(PATH_PNET, allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.Tensor(weights[n])

    def forward(self, x):
        x = self.features(x)

        y_conf = self.conv4_1(x)
        y_bbox = self.conv4_2(x)

        y_conf = F.softmax(y_conf, dim=1)
        
        return y_bbox, y_conf


class RNet(nn.Module):

    def __init__(self):

        super(RNet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 28, 3, 1)),
            ('prelu1', nn.PReLU(28)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(28, 48, 3, 1)),
            ('prelu2', nn.PReLU(48)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', nn.Conv2d(48, 64, 2, 1)),
            ('prelu3', nn.PReLU(64)),

            ('flatten', Flatten()),
            ('conv4', nn.Linear(576, 128)),
            ('prelu4', nn.PReLU(128))
        ]))

        self.conv5_1 = nn.Linear(128, 2)
        self.conv5_2 = nn.Linear(128, 4)

        weights = np.load(PATH_RNET, allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.Tensor(weights[n])

    def forward(self, x):
        x = self.features(x)

        y_conf = self.conv5_1(x)
        y_bbox = self.conv5_2(x)

        y_conf = F.softmax(y_conf, dim=1)

        return y_bbox, y_conf


class ONet(nn.Module):

    def __init__(self):

        super(ONet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 3, 1)),
            ('prelu1', nn.PReLU(32)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(32, 64, 3, 1)),
            ('prelu2', nn.PReLU(64)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', nn.Conv2d(64, 64, 3, 1)),
            ('prelu3', nn.PReLU(64)),
            ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv4', nn.Conv2d(64, 128, 2, 1)),
            ('prelu4', nn.PReLU(128)),

            ('flatten', Flatten()),
            ('conv5', nn.Linear(1152, 256)),
            ('drop5', nn.Dropout(0.25)),
            ('prelu5', nn.PReLU(256)),
        ]))

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)

        weights = np.load(PATH_ONET, allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.Tensor(weights[n])

    def forward(self, x):
        x = self.features(x)

        y_conf = self.conv6_1(x)
        y_bbox = self.conv6_2(x)
        y_land = self.conv6_3(x)

        y_conf = F.softmax(y_conf, dim=1)

        return y_land, y_bbox, y_conf
