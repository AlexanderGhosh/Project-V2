import numpy as np
import cv2 as cv
import matplotlib.pyplot as graph
from sklearn.cluster import KMeans
import time

THRESH_HOLD = 0.7
TARG = 0.0, 0.0, 1.0
CLUSTERS = 2
MAX_ITTER = 10


def how_red(orig: tuple) -> float:
    b1, g1, r1 = orig / 255.0
    r2, g2, b2 = TARG
    t = (r1 - r2)**2.0 + (g1 - g2)**2.0 + (b1 - b2)**2.0
    # t = t**0.5
    t /= 3.0**0.5
    return 1.0 - t

video = cv.VideoCapture('./dataset/3.mp4')
# im = cv.imread('./dataset/2.jpg')

mask = np.zeros((325, int(video.get(cv.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.uint8)
t = mask.shape[1:-1]
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('001.mov', fourcc , 25.0, t[::-1])

height_ = video.get(cv.CAP_PROP_FRAME_HEIGHT)


start_ = time.process_time()
frame_count = -1
while video.isOpened():
    frame_count += 1
    print(f'starting frame: {frame_count+1}')
    suc, frame = video.read()
    if not suc:
        break

    points_ = []
    for x in range(frame.shape[0]):
        for y in range(frame.shape[1]):
            close = how_red(frame[x, y])
            if close >= THRESH_HOLD:
                px = 255
                points_.append((x, y))
            else:
                px = 0
            mask[frame_count, x, y] = [px] * 3
    t = out.write(mask[frame_count])

    if len(points_) == 0:
        continue

    dataset = np.asarray(points_)
    kmeans = KMeans(CLUSTERS).fit(dataset)

    split_ = [[] for i in range(CLUSTERS)]

    maxs_ = [[0, 0] for i in range(CLUSTERS)]
    mins_ = [[100000, 100000] for i in range(CLUSTERS)]

    for (x, y), label in zip(points_, kmeans.labels_):
        split_[label].append((x, y))
        '''maxs_[label][0] = max(maxs_[label][0], x)
        maxs_[label][1] = max(maxs_[label][1], y)

        mins_[label][0] = min(mins_[label][0], x)
        mins_[label][1] = min(mins_[label][1], y)'''

    for c in range(CLUSTERS):
        centre_ = np.floor(kmeans.cluster_centers_[c])
        mask[frame_count] = cv.circle(mask[frame_count], (int(centre_[0]), int(centre_[1])), 10, (0, 0, 125), -1)
        max_ = [0, 0]
        min_ = [10000000, 100000000]
        for point in split_[c]:
            for i in range(2):
                max_[i] = max(max_[i], point[i])
                min_[i] = min(min_[i], point[i])

        mask[frame_count] = cv.rectangle(mask[frame_count], min_, max_, (0, 255, 255))

elapsed_ = time.process_time() - start_
video.release()
out.release()
# bounding boxes drawn
print(f'{elapsed_} seconds elapsed')
'''#cv.imshow('image', im)
cv.startWindowThread()
cv.imshow('mask', mask)

cv.waitKey(0)

cv.destroyAllWindows()'''


'''for index, frame in enumerate(mask):
    cv.imshow(f'Frame: {index}', frame)
    cv.waitKey()
    cv.destroyAllWindows()'''
# graph.imshow(mask[0])
#graph.show()
