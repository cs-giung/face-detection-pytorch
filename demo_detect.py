import time, cv2
from matplotlib import pyplot as plt
from detectors import MTCNN
from detectors import FaceBoxes
from detectors import TinyFace
from detectors import PyramidBox
from detectors import S3FD
from detectors import DSFD
from utils import draw_bboxes


def plot_figures(figures, nrows=1, ncols=1):
    _, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind, title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
    plt.tight_layout()
    plt.show()

# load image with cv in RGB.
IMAGE_PATH = 'selfie.jpg'
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# load detectors.
DET1 = MTCNN(device='cuda')
DET2 = FaceBoxes(device='cuda')
DET3 = TinyFace(device='cuda')
DET4 = PyramidBox(device='cuda')
DET5 = S3FD(device='cuda')
DET6 = DSFD(device='cuda')

# MTCNN returns bboxes and landmarks.
t = time.time()
bboxes, _ = DET1.detect_faces(img, conf_th=0.9, scales=[1])
print('MTCNN : %d faces in %.4f seconds.' % (len(bboxes), time.time() - t))
img1 = draw_bboxes(img, bboxes)
sizes = []
for box in bboxes:
    sizes.append((box[2] - box[0]) * (box[3] - box[1]))
print(min(sizes))
print(max(sizes))

# FaceBoxes returns bboxes.
t = time.time()
bboxes = DET2.detect_faces(img, conf_th=0.9, scales=[1])
print('FaceBoxes : %d faces in %.4f seconds.' % (len(bboxes), time.time() - t))
img2 = draw_bboxes(img, bboxes)
sizes = []
for box in bboxes:
    sizes.append((box[2] - box[0]) * (box[3] - box[1]))
print(min(sizes))
print(max(sizes))

# Tiny Face returns bboxes.
t = time.time()
bboxes = DET3.detect_faces(img, conf_th=0.9, scales=[1])
print('Tiny Face : %d faces in %.4f seconds.' % (len(bboxes), time.time() - t))
img3 = draw_bboxes(img, bboxes)
sizes = []
for box in bboxes:
    sizes.append((box[2] - box[0]) * (box[3] - box[1]))
print(min(sizes))
print(max(sizes))

# PyramidBox returns bboxes.
t = time.time()
bboxes = DET4.detect_faces(img, conf_th=0.9, scales=[1])
print('PyramidBox : %d faces in %.4f seconds.' % (len(bboxes), time.time() - t))
img4 = draw_bboxes(img, bboxes)
sizes = []
for box in bboxes:
    sizes.append((box[2] - box[0]) * (box[3] - box[1]))
print(min(sizes))
print(max(sizes))

# S3FD returns bboxes.
t = time.time()
bboxes = DET5.detect_faces(img, conf_th=0.9, scales=[1])
print('S3FD : %d faces in %.4f seconds.' % (len(bboxes), time.time() - t))
img5 = draw_bboxes(img, bboxes)
sizes = []
for box in bboxes:
    sizes.append((box[2] - box[0]) * (box[3] - box[1]))
print(min(sizes))
print(max(sizes))

# DSFD returns bboxes.
t = time.time()
bboxes = DET6.detect_faces(img, conf_th=0.9, scales=[1])
print('DSFD : %d faces in %.4f seconds.' % (len(bboxes), time.time() - t))
img6 = draw_bboxes(img, bboxes)
sizes = []
for box in bboxes:
    sizes.append((box[2] - box[0]) * (box[3] - box[1]))
print(min(sizes))
print(max(sizes))

# plot results.
results = {
    'MTCNN (conf_th=0.9, scales=[1])': img1,
    'FaceBoxes (conf_th=0.9, scales=[1])': img2,
    'Tiny Face (conf_th=0.9, scales=[1])': img3,
    'PyramidBox (conf_th=0.9, scales=[1])': img4,
    'S3FD (conf_th=0.9, scales=[1])': img5,
    'DSFD (conf_th=0.9, scales=[1])': img6
}
plot_figures(results, 2, 3)
