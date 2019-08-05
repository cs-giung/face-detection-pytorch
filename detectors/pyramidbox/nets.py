import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .box_utils import Detect, PriorBox


class L2Norm(nn.Module):

    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class conv_bn(nn.Module):
    """docstring for conv"""

    def __init__(self,
                 in_plane,
                 out_plane,
                 kernel_size,
                 stride,
                 padding):
        super(conv_bn, self).__init__()
        self.conv1 = nn.Conv2d(in_plane, out_plane, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_plane)

    def forward(self, x):
        x = self.conv1(x)
        return self.bn1(x)


class CPM(nn.Module):

    def __init__(self, in_plane):
        super(CPM, self).__init__()
        self.branch1 = conv_bn(in_plane, 1024, 1, 1, 0)
        self.branch2a = conv_bn(in_plane, 256, 1, 1, 0)
        self.branch2b = conv_bn(256, 256, 3, 1, 1)
        self.branch2c = conv_bn(256, 1024, 1, 1, 0)

        self.ssh_1 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.ssh_dimred = nn.Conv2d(1024, 128, kernel_size=3, stride=1, padding=1)
        self.ssh_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.ssh_3a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.ssh_3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out_residual = self.branch1(x)
        x = F.relu(self.branch2a(x), inplace=True)
        x = F.relu(self.branch2b(x), inplace=True)
        x = self.branch2c(x)

        rescomb = F.relu(x + out_residual, inplace=True)
        ssh1 = self.ssh_1(rescomb)
        ssh_dimred = F.relu(self.ssh_dimred(rescomb), inplace=True)
        ssh_2 = self.ssh_2(ssh_dimred)
        ssh_3a = F.relu(self.ssh_3a(ssh_dimred), inplace=True)
        ssh_3b = self.ssh_3b(ssh_3a)

        ssh_out = torch.cat([ssh1, ssh_2, ssh_3b], dim=1)
        ssh_out = F.relu(ssh_out, inplace=True)
        return ssh_out


class PyramidBoxNet(nn.Module):

    def __init__(self, device='cuda'):
        super(PyramidBoxNet, self).__init__()
        self.device = device

        self.vgg = nn.ModuleList([
            nn.Conv2d(3, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            
            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 1024, 3, 1, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 1, 1),
            nn.ReLU(inplace=True),
        ])

        self.extras = nn.ModuleList([
            nn.Conv2d(1024, 256, 1, 1),
            nn.Conv2d(256, 512, 3, 2, padding=1),
            nn.Conv2d(512, 128, 1, 1),
            nn.Conv2d(128, 256, 3, 2, padding=1)
        ])

        self.L2Norm3_3 = L2Norm(256, 10)
        self.L2Norm4_3 = L2Norm(512, 8)
        self.L2Norm5_3 = L2Norm(512, 5)

        self.lfpn_topdown = nn.ModuleList([
            nn.Conv2d(1024, 512, 1, 1),
            nn.Conv2d(512, 512, 1, 1),
            nn.Conv2d(512, 256, 1, 1)
        ])
        self.lfpn_later = nn.ModuleList([
            nn.Conv2d(512, 512, 1, 1),
            nn.Conv2d(512, 512, 1, 1),
            nn.Conv2d(256, 256, 1, 1)
        ])
        self.cpm = nn.ModuleList([
            CPM(256), CPM(512), CPM(512),
            CPM(1024), CPM(512), CPM(256)
        ])

        self.loc_layers = nn.ModuleList([
            nn.Conv2d(512, 8, 3, 1, padding=1),
            nn.Conv2d(512, 8, 3, 1, padding=1),
            nn.Conv2d(512, 8, 3, 1, padding=1),
            nn.Conv2d(512, 8, 3, 1, padding=1),
            nn.Conv2d(512, 8, 3, 1, padding=1),
            nn.Conv2d(512, 8, 3, 1, padding=1)
        ])

        self.conf_layers = nn.ModuleList([
            nn.Conv2d(512, 8, 3, 1, padding=1),
            nn.Conv2d(512, 6, 3, 1, padding=1),
            nn.Conv2d(512, 6, 3, 1, padding=1),
            nn.Conv2d(512, 6, 3, 1, padding=1),
            nn.Conv2d(512, 6, 3, 1, padding=1),
            nn.Conv2d(512, 6, 3, 1, padding=1)
        ])

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect()

    def _upsample_prod(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) * y

    def forward(self, x):
        size = x.size()[2:]

        for k in range(0, 16):
            x = self.vgg[k](x)
        conv3_3 = x

        for k in range(16, 23):
            x = self.vgg[k](x)
        conv4_3 = x

        for k in range(23, 30):
            x = self.vgg[k](x)
        conv5_3 = x

        for k in range(30, 35):
            x = self.vgg[k](x)
        convfc_7 = x

        for k in range(0, 2):
            x = F.relu(self.extras[k](x), inplace=True)
        conv6_2 = x
        
        for k in range(2, 4):
            x = F.relu(self.extras[k](x), inplace=True)
        conv7_2 = x

        x = F.relu(self.lfpn_topdown[0](convfc_7), inplace=True)
        lfpn2_on_conv5 = F.relu(self._upsample_prod(x, self.lfpn_later[0](conv5_3)), inplace=True)

        x = F.relu(self.lfpn_topdown[1](lfpn2_on_conv5), inplace=True)
        lfpn1_on_conv4 = F.relu(self._upsample_prod(x, self.lfpn_later[1](conv4_3)), inplace=True)

        x = F.relu(self.lfpn_topdown[2](lfpn1_on_conv4), inplace=True)
        lfpn0_on_conv3 = F.relu(self._upsample_prod(x, self.lfpn_later[2](conv3_3)), inplace=True)

        ssh_conv3_norm = self.cpm[0](self.L2Norm3_3(lfpn0_on_conv3))
        ssh_conv4_norm = self.cpm[1](self.L2Norm4_3(lfpn1_on_conv4))
        ssh_conv5_norm = self.cpm[2](self.L2Norm5_3(lfpn2_on_conv5))
        ssh_convfc7 = self.cpm[3](convfc_7)
        ssh_conv6 = self.cpm[4](conv6_2)
        ssh_conv7 = self.cpm[5](conv7_2)

        face_locs, face_confs = [], []
        head_locs, head_confs = [], []

        N = ssh_conv3_norm.size(0)

        mbox_loc = self.loc_layers[0](ssh_conv3_norm)
        face_loc, head_loc = torch.chunk(mbox_loc, 2, dim=1)

        face_loc = face_loc.permute(0, 2, 3, 1).contiguous().view(N, -1, 4)
        mbox_conf = self.conf_layers[0](ssh_conv3_norm)
        face_conf1 = mbox_conf[:, 3:4, :, :]
        face_conf3_maxin, _ = torch.max(mbox_conf[:, 0:3, :, :], dim=1, keepdim=True)

        face_conf = torch.cat((face_conf3_maxin, face_conf1), dim=1)
        face_conf = face_conf.permute(0, 2, 3, 1).contiguous().view(N, -1, 2)

        face_locs.append(face_loc)
        face_confs.append(face_conf)

        inputs = [ssh_conv4_norm, ssh_conv5_norm, ssh_convfc7, ssh_conv6, ssh_conv7]

        feature_maps = []
        feat_size = ssh_conv3_norm.size()[2:]
        feature_maps.append([feat_size[0], feat_size[1]])

        for i, feat in enumerate(inputs):
            feat_size = feat.size()[2:]
            feature_maps.append([feat_size[0], feat_size[1]])
            mbox_loc = self.loc_layers[i + 1](feat)
            face_loc, head_loc = torch.chunk(mbox_loc, 2, dim=1)
            face_loc = face_loc.permute(0, 2, 3, 1).contiguous().view(N, -1, 4)
            mbox_conf = self.conf_layers[i + 1](feat)
            face_conf1 = mbox_conf[:, 0:1, :, :]
            face_conf3_maxin, _ = torch.max(mbox_conf[:, 1:4, :, :], dim=1, keepdim=True)
            face_conf = torch.cat((face_conf1, face_conf3_maxin), dim=1)
            face_conf = face_conf.permute(0, 2, 3, 1).contiguous().view(N, -1, 2)
            face_locs.append(face_loc)
            face_confs.append(face_conf)

        face_mbox_loc = torch.cat(face_locs, dim=1)
        face_mbox_conf = torch.cat(face_confs, dim=1)

        with torch.no_grad():
            self.priors_boxes = PriorBox(size, feature_maps)
            self.priors = self.priors_boxes.forward()

        output = self.detect(
            face_mbox_loc,
            self.softmax(face_mbox_conf),
            self.priors.to(self.device)
        )

        return output
