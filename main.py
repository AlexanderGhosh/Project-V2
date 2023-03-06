import numpy as np
import cv2 as cv
import matplotlib.pyplot as graph

im = cv.imread('./dataset/1.jpg')
im = cv.cvtColor(im, cv.COLOR_BGR2RGB)

graph.imshow(im)
graph.show()