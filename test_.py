import time, cv2
from matplotlib import pyplot as plt
from detectors import PyramidBox, FaceBoxes
from utils import draw_bboxes


def plot_figures(figures, nrows=1, ncols=1):
    _, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind, title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
    plt.tight_layout()
    plt.show()

# load image with cv in RGB.
IMAGE_PATH = 'bts.jpg'
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# load detectors.
DET1 = PyramidBox(device='cpu')
DET2 = FaceBoxes(device='cpu')

t = time.time()
bboxes = DET1.detect_faces(img, conf_th=0.9, scales=[1])
print('%d faces in %.4f seconds.' % (len(bboxes), time.time() - t))
img1 = draw_bboxes(img, bboxes, thickness=3)

t = time.time()
bboxes = DET2.detect_faces(img, conf_th=0.9, scales=[1])
print('%d faces in %.4f seconds.' % (len(bboxes), time.time() - t))
img2 = draw_bboxes(img, bboxes, thickness=3)

# plot results.
results = {
    'PyramidBox': img1,
    'FaceBoxes': img2,
}
plot_figures(results, 1, 2)
