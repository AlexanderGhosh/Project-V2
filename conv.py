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

device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('cuda')
else:
    device = torch.device('cpu')
    print('cpu')

im = cv.imread('./dataset/1.jpg')
im = cv.cvtColor(im, cv.COLOR_BGR2RGB).astype(np.float32)

n = np.sum(im, axis=2)

n = np.repeat(n[:,:,np.newaxis], 3, axis=2)
im /= n

im = im.T
im = np.asarray([im]).astype(np.float32)


w_ = np.asarray([TARGET_COLOUR])
w_ = np.expand_dims(w_, axis=-1)
w_ = np.expand_dims(w_, axis=-1)

w_ = torch.asarray(w_.astype(np.float32))

conv = nn.Conv2d(3, 1, 1)
w_ = nn.Parameter(w_)
conv.weight = w_
conv.bias = nn.Parameter(torch.zeros((1,)))
conv = conv.to(device)

thresh = Threshold(0.4)
thresh = thresh.to(device)

x = torch.from_numpy(im)
x = x.to(device)

y = conv(x)
y = thresh(y)

y = y.detach().cpu().numpy()
plot.imshow(y[0].T, cmap='gray')
plot.show()

