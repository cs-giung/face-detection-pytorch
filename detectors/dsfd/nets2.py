from math import sqrt as sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision
from .box_utils import decode, nms


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


class PriorBox(object):

    def __init__(self, min_size, max_size, feature_maps, image_size):
        super(PriorBox, self).__init__()
        self.image_size = image_size
        self.feature_maps = feature_maps
        self.num_priors = 6
        self.variance = [0.1, 0.2] or [0.1]
        self.min_sizes = min_size
        self.max_sizes = max_size
        self.steps = [4, 8, 16, 32, 64, 128]
        self.aspect_ratios = [ [1.5],[1.5],[1.5],[1.5],[1.5],[1.5] ]
        self.clip = True

    def forward(self):
        mean = []
        if len(self.min_sizes) == 5:
            self.feature_maps = self.feature_maps[1:]
            self.steps = self.steps[1:]
        if len(self.min_sizes) == 4:
            self.feature_maps = self.feature_maps[2:]
            self.steps = self.steps[2:]

        for k, f in enumerate(self.feature_maps):
            #for i, j in product(range(f), repeat=2):
            for i in range(f[0]):
                for j in range(f[1]):
                    #f_k = self.image_size / self.steps[k]
                    f_k_i = self.image_size[0] / self.steps[k]
                    f_k_j = self.image_size[1] / self.steps[k]
                    # unit center x,y
                    cx = (j + 0.5) / f_k_j
                    cy = (i + 0.5) / f_k_i
                    # aspect_ratio: 1
                    # rel size: min_size
                    s_k_i = self.min_sizes[k]/self.image_size[1]
                    s_k_j = self.min_sizes[k]/self.image_size[0]
                    # swordli@tencent
                    if len(self.aspect_ratios[0]) == 0:
                        mean += [cx, cy, s_k_i, s_k_j]

                    # aspect_ratio: 1
                    # rel size: sqrt(s_k * s_(k+1))
                    #s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                    if len(self.max_sizes) == len(self.min_sizes):
                        s_k_prime_i = sqrt(s_k_i * (self.max_sizes[k]/self.image_size[1]))
                        s_k_prime_j = sqrt(s_k_j * (self.max_sizes[k]/self.image_size[0]))    
                        mean += [cx, cy, s_k_prime_i, s_k_prime_j]
                    # rest of aspect ratios
                    for ar in self.aspect_ratios[k]:
                        if len(self.max_sizes) == len(self.min_sizes):
                            mean += [cx, cy, s_k_prime_i/sqrt(ar), s_k_prime_j*sqrt(ar)]
                        mean += [cx, cy, s_k_i/sqrt(ar), s_k_j*sqrt(ar)]
                
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, top_k=5000, conf_thresh=0.01, nms_thresh=0.3):
        self.num_classes = 2
        self.background_label = 0
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.variance = [0.1, 0.2]

    def forward(self, loc_data, conf_data, prior_data, arm_loc_data=None , arm_conf_data=None):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        
        #swordli
        #num_priors = loc_data.size(1)
       
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,  self.num_classes).transpose(2, 1)
        
        # Decode predictions into bboxes.
        for i in range(num):
            default = prior_data
            decoded_boxes = decode(loc_data[i], default, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output


class DSFDNet2(nn.Module):

    def __init__(self, device='cuda'):
        super(DSFDNet2, self).__init__()
        self.device = device

        resnet = torchvision.models.resnet152(pretrained=True)

        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)
        self.layer4 = nn.Sequential(resnet.layer4)
        self.layer5 = nn.Sequential(*[
            nn.Conv2d(2048, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        ])
        self.layer6 = nn.Sequential(*[
            nn.Conv2d(512, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        ])

        self.latlayer3 = nn.Conv2d(2048, 1024, 1, 1, padding=0)
        self.latlayer2 = nn.Conv2d(1024, 512, 1, 1, padding=0)
        self.latlayer1 = nn.Conv2d(512, 256, 1, 1, padding=0)

        self.smooth3 = nn.Conv2d(1024, 1024, 1, 1, padding=0)
        self.smooth2 = nn.Conv2d(512, 512, 1, 1, padding=0)
        self.smooth1 = nn.Conv2d(256, 256, 1, 1, padding=0)

        self.cpm3_3 = FEM(256)
        self.cpm4_3 = FEM(512)
        self.cpm5_3 = FEM(1024)
        self.cpm7 = FEM(2048)
        self.cpm6_2 = FEM(512)
        self.cpm7_2 = FEM(256)

        output_channels = [256, 512, 1024, 2048, 512, 256]
        mbox = [1, 1, 1, 1, 1, 1]
        head = pa_multibox(output_channels, mbox)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect()

    def _upsample_product(self, x, y):
        _, _, H, W = y.size()
        # return F.upsample(x, size=(H, W), mode='bilinear') * y
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) * y

    def mio_module(self, each_mmbox, len_conf):
        chunk = torch.chunk(each_mmbox, each_mmbox.shape[1], 1)
        bmax = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
        cls = (torch.cat([bmax, chunk[3]], dim=1) if len_conf == 0 else torch.cat([chunk[3], bmax], dim=1))
        if len(chunk) == 6:
            cls = torch.cat([cls, chunk[4], chunk[5]], dim=1) 
        elif len(chunk) == 8:
            cls = torch.cat([cls, chunk[4], chunk[5], chunk[6], chunk[7]], dim=1) 
        return cls

    def init_priors(self, min_size=[16, 32, 64, 128, 256, 512], max_size=[], feature_maps=[160, 80, 40, 20, 10, 5], image_size=640):
        priorbox = PriorBox(min_size, max_size, feature_maps, image_size)
        with torch.no_grad():
            prior = priorbox.forward()
        return prior

    def forward(self, x):
        image_size = [x.shape[2], x.shape[3]]
        loc = list()
        conf = list()

        conv3_3_x = self.layer1(x)
        conv4_3_x = self.layer2(conv3_3_x)
        conv5_3_x = self.layer3(conv4_3_x)
        fc7_x = self.layer4(conv5_3_x)
        conv6_2_x = self.layer5(fc7_x)
        conv7_2_x = self.layer6(conv6_2_x)

        lfpn3 = self._upsample_product(self.latlayer3(fc7_x), self.smooth3(conv5_3_x))
        lfpn2 = self._upsample_product(self.latlayer2(lfpn3), self.smooth2(conv4_3_x))
        lfpn1 = self._upsample_product(self.latlayer1(lfpn2), self.smooth1(conv3_3_x))
        conv5_3_x = lfpn3
        conv4_3_x = lfpn2
        conv3_3_x = lfpn1

        sources = [conv3_3_x, conv4_3_x, conv5_3_x, fc7_x, conv6_2_x, conv7_2_x]
        sources[0] = self.cpm3_3(sources[0])
        sources[1] = self.cpm4_3(sources[1])
        sources[2] = self.cpm5_3(sources[2])
        sources[3] = self.cpm7(sources[3])
        sources[4] = self.cpm6_2(sources[4])
        sources[5] = self.cpm7_2(sources[5])

        featuremap_size = []
        for (x, l, c) in zip(sources, self.loc, self.conf):
            featuremap_size.append([x.shape[2], x.shape[3]])
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())

            len_conf = len(conf)
            cls = self.mio_module(c(x), len_conf)
            conf.append(cls.permute(0, 2, 3, 1).contiguous())
        
        mbox_num = 1
        face_loc = torch.cat([o[:, :, :, :4].contiguous().view(o.size(0), -1) for o in loc], 1)
        face_conf = torch.cat([o[:, :, :, :2].contiguous().view(o.size(0), -1) for o in conf], 1)
        head_loc = torch.cat([o[:, :, :, 4:8].contiguous().view(o.size(0), -1) for o in loc[1:]], 1)
        head_conf = torch.cat([o[:, :,:, 2:4].contiguous().view(o.size(0), -1) for o in conf[1:]], 1)
        body_loc = torch.cat([o[:, :, :, 8:].contiguous().view(o.size(0), -1) for o in loc[2:]], 1)
        body_conf = torch.cat([o[:, :, :, 4:].contiguous().view(o.size(0), -1) for o in conf[2:]], 1)
        self.priors = self.init_priors(feature_maps=featuremap_size, image_size=image_size)

        output = self.detect(
            face_loc.view(face_loc.size(0), -1, 4),
            self.softmax(face_conf.view(face_conf.size(0), -1, 2)),
            self.priors.type(type(x.data)).to(self.device)
        )

        return output

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
        elif k == 1:
            loc_output = 8
            conf_output = 4
        else:
            loc_output = 12
            conf_output = 6
        loc_layers += [DeepHeadModule(input_channels, mbox_cfg[k] * loc_output)]
        conf_layers += [DeepHeadModule(input_channels, mbox_cfg[k] * (2+conf_output))]
    return (loc_layers, conf_layers)