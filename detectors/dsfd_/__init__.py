import time
import numpy as np
import cv2
import torch
from torchvision import transforms
from .nets import DSFDNet
from .box_utils import nms_

PATH_WEIGHT = './detectors/dsfd_/weights/WIDERFace_DSFD_RES152.pth'
img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')


class DSFD():

    def __init__(self, device='cuda'):

        tstamp = time.time()
        self.device = device

        print('[DSFD] loading with', self.device)
        self.net = DSFDNet(device=self.device).to(self.device)
        state_dict = torch.load(PATH_WEIGHT, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        print('[DSFD] finished loading (%.4f sec)' % (time.time() - tstamp))
    
    def detect_faces(self, image, conf_th=0.8, scales=[1]):
        width, height = image.shape[1], image.shape[0]
        scale = torch.Tensor([width, height, width, height])

        
        for s in scales:
            x = cv2.resize(image, None, None, fx=s, fy=s)
            x = x.astype(np.float32)
            x -= np.array([104, 117, 123], dtype=np.float32)
            x = torch.from_numpy(x).permute(2, 0, 1)
            x = x.unsqueeze(0)
            x = x.to(self.device)
            y = self.net(x)
        return []
