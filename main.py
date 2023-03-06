import numpy as np
import cv2 as cv
import matplotlib.pyplot as graph


def how_red(orig: tuple) -> float:
    b1, g1, r1 = orig
    t = (r1 + (1.0 - g1) + (1.0 - b1)) * 0.333
    return t


im = cv.imread('./dataset/1.jpg')

mask = np.zeros((im.shape[:-1]), dtype=np.uint8)

THRESH_HOLD = 0.7
for x in range(im.shape[0]):
    for y in range(im.shape[1]):
        close = how_red(im[x, y])
        if close >= THRESH_HOLD:
            px = 255
        else:
            px = 0
        mask[x, y] = px


cv.imshow('image', im)
cv.startWindowThread()
cv.imshow('mask', mask)

cv.waitKey(0)

cv.destroyAllWindows()
cv.waitKey(1)
cv.waitKey(1)
cv.waitKey(1)
cv.waitKey(1)


