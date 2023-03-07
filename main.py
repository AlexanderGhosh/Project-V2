import numpy as np
import cv2 as cv
import matplotlib.pyplot as graph

THRESH_HOLD = 0.7
TARG = 0.0, 1.0, 0.0
def how_red(orig: tuple) -> float:
    b1, g1, r1 = orig / 255.0
    r2, g2, b2 = TARG
    t = (r1 - r2)**2.0 + (g1 - g2)**2.0 + (b1 - b2)**2.0
    # t = t**0.5
    t /= 3.0**0.5
    return 1.0 - t


im = cv.imread('./dataset/1.jpg')

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

mean_ = sum(xs_) // len(xs_), sum(ys_) // len(xs_)
mask = cv.circle(mask, mean_, 5, (255, 125, 125), -1)

p = [(max(xs_), max(ys_)), (min(xs_), min(ys_))]

print(p)
mask = cv.rectangle(mask, p[0], p[1], (0, 255, 255))

'''#cv.imshow('image', im)
cv.startWindowThread()
cv.imshow('mask', mask)

cv.waitKey(0)

cv.destroyAllWindows()'''


graph.imshow(mask)
graph.show()
