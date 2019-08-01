import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .box_utils import Detect, PriorBox


class FEM(nn.Module):

    def __init__(self, channel_size):
        super(FEM , self).__init__()
        self.cs = channel_size
        self.cpm1 = nn.Conv2d( self.cs, 256, kernel_size=3, dilation=1, stride=1, padding=1)
        self.cpm2 = nn.Conv2d( self.cs, 256, kernel_size=3, dilation=2, stride=1, padding=2)
        self.cpm3 = nn.Conv2d( 256, 128, kernel_size=3, dilation=1, stride=1, padding=1)
        self.cpm4 = nn.Conv2d( 256, 128, kernel_size=3, dilation=2, stride=1, padding=2)
        self.cpm5 = nn.Conv2d( 128, 128, kernel_size=3, dilation=1, stride=1, padding=1)

    def forward(self, x):
        x1_1 = F.relu(self.cpm1(x), inplace=True)
        x1_2 = F.relu(self.cpm2(x), inplace=True)
        x2_1 = F.relu(self.cpm3(x1_2), inplace=True)
        x2_2 = F.relu(self.cpm4(x1_2), inplace=True)
        x3_1 = F.relu(self.cpm5(x2_2), inplace=True)
        return torch.cat([x1_1, x2_1, x3_1] , 1)


class DeepHeadModule(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(DeepHeadModule , self).__init__()
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._mid_channels = min(self._input_channels, 256)
        self.conv1 = nn.Conv2d( self._input_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv2 = nn.Conv2d( self._mid_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv3 = nn.Conv2d( self._mid_channels, self._mid_channels, kernel_size=3, dilation=1, stride=1, padding=1)
        self.conv4 = nn.Conv2d( self._mid_channels, self._output_channels, kernel_size=1, dilation=1, stride=1, padding=0)
        
    def forward(self, x):
        return self.conv4(F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1(x), inplace=True)), inplace=True)), inplace=True))


def pa_multibox(output_channels, mbox_cfg):
    loc_layers = []
    conf_layers = []
    for k, v in enumerate(output_channels):
        input_channels = 512
        if k == 0:
            loc_output = 4
            conf_output = 2
        elif k==1:
            loc_output = 8
            conf_output = 4
        else:
            loc_output = 12
            conf_output = 6
        loc_layers += [DeepHeadModule(input_channels, mbox_cfg[k] * loc_output)]
        conf_layers += [DeepHeadModule(input_channels, mbox_cfg[k] * (2+conf_output))]
    return (loc_layers, conf_layers)


class DSFDNet(nn.Module):

    def __init__(self, device='cuda'):
        super(DSFDNet, self).__init__()
        self.device = device

        resnet = torchvision.models.resnet152(pretrained=True)
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)
        self.layer5 = nn.Sequential(                                      
               *[nn.Conv2d(2048, 512, kernel_size=1),                         
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(512,512, kernel_size=3,padding=1,stride=2),
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True)]
        )
        self.layer6 = nn.Sequential(
               *[nn.Conv2d(512, 128, kernel_size=1,),
                  nn.BatchNorm2d(128),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(128, 256, kernel_size=3,padding=1,stride=2),
                  nn.BatchNorm2d(256),
                  nn.ReLU(inplace=True)]
        )

        output_channels = [256, 512, 1024, 2048, 512, 256]

        # feature_pyramid_network
        fpn_in = output_channels
        self.latlayer3 = nn.Conv2d( fpn_in[3], fpn_in[2], kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( fpn_in[2], fpn_in[1], kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d( fpn_in[1], fpn_in[0], kernel_size=1, stride=1, padding=0)
        self.smooth3 = nn.Conv2d( fpn_in[2], fpn_in[2], kernel_size=1, stride=1, padding=0)
        self.smooth2 = nn.Conv2d( fpn_in[1], fpn_in[1], kernel_size=1, stride=1, padding=0)
        self.smooth1 = nn.Conv2d( fpn_in[0], fpn_in[0], kernel_size=1, stride=1, padding=0)

        # feature_enhance_module
        cpm_in = output_channels
        self.cpm3_3 = FEM(cpm_in[0])
        self.cpm4_3 = FEM(cpm_in[1])
        self.cpm5_3 = FEM(cpm_in[2])
        self.cpm7 = FEM(cpm_in[3])
        self.cpm6_2 = FEM(cpm_in[4])
        self.cpm7_2 = FEM(cpm_in[5])

        # progressive_anchor
        head = pa_multibox(output_channels, [1, 1, 1, 1, 1, 1])
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect()

    def _upsample_product(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) * y

    def mio_module(self, each_mmbox, len_conf):
        chunk = torch.chunk(each_mmbox, each_mmbox.shape[1], 1)
        bmax  = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
        cls = ( torch.cat([bmax,chunk[3]], dim=1) if len_conf==0 else torch.cat([chunk[3],bmax],dim=1) )
        if len(chunk)==6:
            cls = torch.cat([cls, chunk[4], chunk[5]], dim=1) 
        elif len(chunk)==8:
            cls = torch.cat([cls, chunk[4], chunk[5], chunk[6], chunk[7]], dim=1) 
        return cls 

    def forward(self, x):
        image_size = [x.shape[2] , x.shape[3]]
        loc = list()
        conf = list()

        # resnet152
        conv3_3_x = self.layer1(x)
        conv4_3_x = self.layer2(conv3_3_x)
        conv5_3_x = self.layer3(conv4_3_x)
        fc7_x = self.layer4(conv5_3_x)
        conv6_2_x = self.layer5(fc7_x)
        conv7_2_x = self.layer6(conv6_2_x)

        # feature_pyramid_network
        lfpn3 = self._upsample_product( self.latlayer3(fc7_x) , self.smooth3(conv5_3_x) )
        lfpn2 = self._upsample_product( self.latlayer2(lfpn3) , self.smooth2(conv4_3_x) )
        lfpn1 = self._upsample_product( self.latlayer1(lfpn2) , self.smooth1(conv3_3_x) )
        conv5_3_x = lfpn3
        conv4_3_x = lfpn2
        conv3_3_x = lfpn1

        # feature_enhance_module
        sources = [conv3_3_x, conv4_3_x, conv5_3_x, fc7_x, conv6_2_x, conv7_2_x]
        sources[0] = self.cpm3_3(sources[0])
        sources[1] = self.cpm4_3(sources[1])
        sources[2] = self.cpm5_3(sources[2])
        sources[3] = self.cpm7(sources[3])
        sources[4] = self.cpm6_2(sources[4])
        sources[5] = self.cpm7_2(sources[5])

        # featuremap_size = []
        # for (x, l, c) in zip(sources, self.loc, self.conf):
        #     featuremap_size.append([x.shape[2], x.shape[3]])

        #     loc.append(l(x).permute(0, 2, 3, 1).contiguous())

        #     cls = self.mio_module(c(x), len(conf))
        #     conf.append(cls.permute(0, 2, 3, 1).contiguous())

        # face_loc = torch.cat(  [o[:,:,:,:4].contiguous().view(o.size(0),-1) for o in loc], 1)
        # face_conf = torch.cat( [o[:,:,:,:2].contiguous().view(o.size(0),-1) for o in conf], 1)


        # with torch.no_grad():
        #     self.priorbox = PriorBox(image_size, featuremap_size)
        #     self.priors = self.priorbox.forward()

        # output = self.detect(
        #     face_loc.view(face_loc.size(0), -1, 4),
        #     self.softmax(face_conf.view(face_conf.size(0), -1, 2)),
        #     self.priors.type(type(x.data)).to(self.device)
        # )
