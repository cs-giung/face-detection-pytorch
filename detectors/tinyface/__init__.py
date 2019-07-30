"""
Finding Tiny Faces
"""

import time
from collections import OrderedDict
import cv2
import numpy as np
import torch
from torchvision import transforms
from .nets import TFNet
from .box_utils import nms, templates


PATH_TFNET = './detectors/tinyface/weights/checkpoint_50.pth'


class TinyFace():

    def __init__(self, device='cuda'):
        tstamp = time.time()
        self.device = device

        print('[Tiny Face] loading with', self.device)
        self.net = TFNet().to(device)

        state_dict = torch.load(PATH_TFNET, map_location=self.device)['model']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)

        self.net.eval()
        print('[Tiny Face] finished loading (%.4f sec)' % (time.time() - tstamp))


    def detect_faces(self, image, conf_th=0.8, scales=[1]):
        """
        input
            image: cv image in RGB.
            conf_th: confidence threshold.
        """

        # image pyramid
        # scales = [0.25, 0.5, 1, 2]

        trans_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # initialize output
        bboxes = np.empty(shape=(0, 5))

        with torch.no_grad():
            for s in scales:

                # input image
                scaled_img = cv2.resize(image, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                img = trans_norm(scaled_img)
                img.unsqueeze_(0)

                # run
                x = img.float().to(self.device)
                y = self.net(x)

                # collect scores
                score_cls = y[:, :25, :, :]
                prob_cls = torch.sigmoid(score_cls)
                score_reg = y[:, 25:125, :, :]
                score_cls = score_cls.data.cpu().numpy().transpose((0, 2, 3, 1))
                prob_cls = prob_cls.data.cpu().numpy().transpose((0, 2, 3, 1))
                score_reg = score_reg.data.cpu().numpy().transpose((0, 2, 3, 1))

                # ignore templates by scale
                tids = list(range(4, 12)) + ([] if s <= 1.0 else list(range(18, 25)))
                ignored_tids = list(set(range(0, 25)) - set(tids))
                try:
                    prob_cls[:, :, ignored_tids] = 0.0
                except IndexError:
                    pass

                # threshold for detection
                indices = np.where(prob_cls > conf_th)
                fb, fy, fx, fc = indices
                scores = prob_cls[fb, fy, fx, fc]
                scores = scores.reshape((scores.shape[0], 1))

                # interpret heatmap into bounding boxes
                cx = fx * 8 - 1
                cy = fy * 8 - 1
                cw = templates[fc, 2] - templates[fc, 0] + 1
                ch = templates[fc, 3] - templates[fc, 1] + 1

                # extract bounding box refinement
                tx = score_reg[0, :, :, 0:25]
                ty = score_reg[0, :, :, 25:50]
                tw = score_reg[0, :, :, 50:75]
                th = score_reg[0, :, :, 75:100]

                # refine bounding boxes
                dcx = cw * tx[fy, fx, fc]
                dcy = ch * ty[fy, fx, fc]
                rcx = cx + dcx
                rcy = cy + dcy
                rcw = cw * np.exp(tw[fy, fx, fc])
                rch = ch * np.exp(th[fy, fx, fc])

                # create bbox array
                rcx = rcx.reshape((rcx.shape[0], 1))
                rcy = rcy.reshape((rcy.shape[0], 1))
                rcw = rcw.reshape((rcw.shape[0], 1))
                rch = rch.reshape((rch.shape[0], 1))
                tmp_bboxes = np.array([rcx - rcw / 2, rcy - rch / 2, rcx + rcw / 2, rcy + rch / 2])
                tmp_bboxes = tmp_bboxes * (1 / s)

                # bboxes with confidences
                for i in range(tmp_bboxes.shape[1]):
                    bbox = (tmp_bboxes[0][i][0], tmp_bboxes[1][i][0], tmp_bboxes[2][i][0], tmp_bboxes[3][i][0], scores[i][0])
                    bboxes = np.vstack((bboxes, bbox))

            # nms
            keep = nms(bboxes, 0.1)
            bboxes = bboxes[keep]

        return bboxes
