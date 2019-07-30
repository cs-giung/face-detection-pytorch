import time
from collections import OrderedDict
import numpy as np
import cv2
import torch
from torchvision import transforms
from .nets import FaceBoxesNet
from .box_utils import PriorBox, decode, nms_


PATH_WEIGHT = './detectors/faceboxes/weights/FaceBoxes.pth'


class FaceBoxes():

    def __init__(self, device='cuda'):

        tstamp = time.time()
        self.device = device

        print('[FaceBoxes] loading with', self.device)
        self.net = FaceBoxesNet().to(self.device)

        state_dict = torch.load(PATH_WEIGHT, map_location=self.device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)

        self.net.eval()
        print('[FaceBoxes] finished loading (%.4f sec)' % (time.time() - tstamp))

    def detect_faces(self, image, conf_th=0.8, scales=[1]):

        bboxes = np.empty(shape=(0, 5))

        for s in scales:
            img = cv2.resize(image, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            img = np.float32(img)
            img_width, img_height = img.shape[1], img.shape[0]
            img -= (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.to(self.device)
            scale = torch.Tensor([img_width, img_height, img_width, img_height])
            scale = scale.to(self.device)

            loc, conf = self.net(img)

            priorbox = PriorBox(image_size=(img_height, img_width))
            priors = priorbox.forward()
            priors = priors.to(self.device)
            prior_data = priors.data

            boxes = decode(loc.data.squeeze(0), prior_data, [0.1, 0.2])
            boxes = boxes * scale / s
            boxes = boxes.cpu().numpy()
            scores = conf.data.cpu().numpy()[:, 1]

            inds = np.where(scores > conf_th)[0]
            boxes = boxes[inds]
            scores = scores[inds]

            order = scores.argsort()[::-1][:5000]
            boxes = boxes[order]
            scores = scores[order]

            dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = nms_(dets, 0.1)
            dets = dets[keep, :]
            dets = dets[:750, :]

            for bbox in dets:
                bboxes = np.vstack((bboxes, bbox))

        keep = nms_(bboxes, 0.1)
        bboxes = bboxes[keep]

        return bboxes
