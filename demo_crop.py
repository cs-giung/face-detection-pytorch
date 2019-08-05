import time, cv2
from matplotlib import pyplot as plt
from detectors import DSFD
from utils import crop_thumbnail


# load image with cv in RGB.
IMAGE_PATH = 'BTS.jpg'
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# load detector.
DET = DSFD(device='cuda')

# DSFD returns bboxes.
t = time.time()
bboxes = DET.detect_faces(img, conf_th=0.95)
print('detect %d faces in %.4f seconds.' % (len(bboxes), time.time() - t))

# crop thumbnail from original image.
results = dict()
t = time.time()
for i, bbox in enumerate(bboxes):
    thumb_img, _ = crop_thumbnail(img, bbox, padding=1, size=100)
    results[str(i)] = thumb_img
print('crop %d faces in %.4f seconds.' % (len(results), time.time() - t))

# plot results
grid = plt.GridSpec(2, len(results))
plt.subplot(grid[0, 0:]).imshow(img)
for i in range(len(results)):
    plt.subplot(grid[1, i]).imshow(results[str(i)])
plt.show()
