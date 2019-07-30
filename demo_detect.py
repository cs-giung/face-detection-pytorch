import time, cv2
from matplotlib import pyplot as plt
from detectors import MTCNN
from detectors import TinyFace
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
DET1 = MTCNN(device='cuda:1')
DET2 = TinyFace(device='cuda:2')
DET3 = S3FD(device='cuda:3')
DET4 = DSFD(device='cuda:4')

# MTCNN returns bboxes and landmarks.
t = time.time()
bboxes, _ = DET1.detect_faces(img, conf_th=0.9, scales=[0.125, 0.25, 0.5, 1])
print('MTCNN : %d faces in %.4f seconds.' % (len(bboxes), time.time() - t))
img1 = draw_bboxes(img, bboxes)

# Tiny Face returns bboxes.
t = time.time()
bboxes = DET2.detect_faces(img, conf_th=0.9, scales=[1])
print('Tiny Face : %d faces in %.4f seconds.' % (len(bboxes), time.time() - t))
img2 = draw_bboxes(img, bboxes)

# S3FD returns bboxes.
t = time.time()
bboxes = DET3.detect_faces(img, conf_th=0.9, scales=[0.5, 1])
print('S3FD : %d faces in %.4f seconds.' % (len(bboxes), time.time() - t))
img3 = draw_bboxes(img, bboxes)

# DSFD returns bboxes.
t = time.time()
bboxes = DET4.detect_faces(img, conf_th=0.9, scales=[0.5, 1])
print('DSFD : %d faces in %.4f seconds.' % (len(bboxes), time.time() - t))
img4 = draw_bboxes(img, bboxes)

# plot results.
results = {
    'MTCNN (conf_th=0.9, scales=[0.125])': img1,
    'Tiny Face (conf_th=0.9, scales=[1])': img2,
    'S3FD (conf_th=0.9, scales=[0.5])': img3,
    'DSFD (conf_th=0.9, scales=[0.5])': img4
}
plot_figures(results, 2, 2)
