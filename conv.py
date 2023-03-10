import matplotlib.pyplot as plot
import torch
from torch import nn
import numpy as np
import cv2 as cv


class Threshold(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        return torch.ceil(x - self.threshold - 0.001)


TARGET_COLOUR = [1.0, 0.0, 0.0]


im = cv.imread('./dataset/1.jpg')
im = cv.cvtColor(im, cv.COLOR_BGR2RGB)

im = (im.T / 255.0).astype(np.float32)
im = np.asarray([im])

conv = nn.Conv2d(3, 1, 1)

w_ = np.asarray([TARGET_COLOUR])
w_ = np.expand_dims(w_, axis=-1)
w_ = np.expand_dims(w_, axis=-1)
w_ /= np.sum(TARGET_COLOUR)

w_ = torch.asarray(w_.astype(np.float32))

w_ = nn.Parameter(w_)
conv.weight = w_
conv.bias = nn.Parameter(torch.zeros((1,)))

thesh = Threshold(0.4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.from_numpy(im)
#x.to(device)
#conv.to(device)

y = conv(x)
y = thesh(y)
#y.to('cpu')
y = y.detach().numpy()

plot.imshow(y[0].T, cmap='gray')
plot.show()

