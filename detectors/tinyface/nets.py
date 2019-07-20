import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet101


class TFNet(nn.Module):

    def __init__(self):

        super(TFNet, self).__init__()

        self.model = resnet101(pretrained=True)
        del self.model.layer4

        self.score_res3 = nn.Conv2d(512, 125, 1, 1)
        self.score_res4 = nn.Conv2d(1024, 125, 1, 1)

        self.score4_upsample = nn.ConvTranspose2d(125, 125, 4, 2, padding=1, bias=False)

        self._init_bilinear()

    def _init_bilinear(self):
        k_size = self.score4_upsample.kernel_size[0]
        ch_in = self.score4_upsample.in_channels
        ch_out = self.score4_upsample.out_channels

        factor = np.floor((k_size + 1) / 2)
        if k_size % 2 == 1:
            center = factor
        else:
            center = factor + 0.5
        
        f = np.zeros((ch_in, ch_out, k_size, k_size))
        C = np.array([1, 2, 3, 4])

        temp = (np.ones((1, k_size)) - (np.abs(C - center) / factor))
        for i in range(ch_out):
            f[i, i, :, :] = temp.T @ temp
        
        self.score4_upsample.weight = torch.nn.Parameter(data=torch.Tensor(f))
    
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        score_res3 = self.score_res3(x)
        x = self.model.layer3(x)
        score_res4 = self.score_res4(x)
        score4 = self.score4_upsample(score_res4)

        cropv = score4.size(2) - score_res3.size(2)
        cropu = score4.size(3) - score_res3.size(3)

        if cropv == 0:
            cropv = -score4.size(2)
        if cropu == 0:
            cropu = -score4.size(3)

        score4 = score4[:, :, 0:-cropv, 0:-cropu]

        score = score_res3 + score4

        return score
