import numpy as np
import cv2 as cv
import matplotlib.pyplot as graph
from sklearn.cluster import KMeans
import time

THRESH_HOLD = 0.7
TARG = 1.0, 0.0, 0.0
CLUSTERS = 2
def how_red(orig: tuple) -> float:
    b1, g1, r1 = orig / 255.0
    r2, g2, b2 = TARG
    t = (r1 - r2)**2.0 + (g1 - g2)**2.0 + (b1 - b2)**2.0
    # t = t**0.5
    t /= 3.0**0.5
    return 1.0 - t


im = cv.imread('./dataset/2.jpg')

mask = np.zeros(im.shape, dtype=np.uint8)


sum_ = [0, 0]
count_ = 0
xs_ = []
ys_ = []

height_ = im.shape[0]
for x in range(im.shape[0]):
    for y in range(im.shape[1]):
        close = how_red(im[x, y])
        if close >= THRESH_HOLD:
            px = 255
            xs_.append(y)
            ys_.append(x)
        else:
            px = 0
        mask[x, y] = [px] * 3

dataset = np.asarray(list(zip(xs_, ys_)))
kmeans = KMeans(CLUSTERS).fit(dataset)


# mean_ = sum(xs_) // len(xs_), sum(ys_) // len(xs_)

'''mean_ = np.floor(kmeans.cluster_centers_[0])
mask = cv.circle(mask, (int(mean_[0]), int(mean_[1])), 10, (0, 0, 125), -1)

mean_ = np.floor(kmeans.cluster_centers_[1])
mask = cv.circle(mask, (int(mean_[0]), int(mean_[1])), 10, (255, 0, 0), -1)
'''
split_ = [[] for i in range(CLUSTERS)]

for point, label in zip(zip(xs_, ys_), kmeans.labels_):
    split_[label].append(point)

for c in range(CLUSTERS):
    centre_ = np.floor(kmeans.cluster_centers_[c])
    mask = cv.circle(mask, (int(centre_[0]), int(centre_[1])), 10, (0, 0, 125), -1)
    max_ = [0, 0]
    min_ = [10000000, 100000000]
    for point in split_[c]:
        for i in range(2):
            max_[i] = max(max_[i], point[i])
            min_[i] = min(min_[i], point[i])

    mask = cv.rectangle(mask, min_, max_, (0, 255, 255))


'''#cv.imshow('image', im)
cv.startWindowThread()
cv.imshow('mask', mask)

cv.waitKey(0)

cv.destroyAllWindows()'''


graph.imshow(mask)
graph.show()
