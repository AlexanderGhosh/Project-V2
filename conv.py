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
        return torch.abs(torch.ceil(x - self.threshold - 0.001))

MAX_FRAMES = 100
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

video = cv.VideoCapture('./dataset/3.mp4')
# frame count, channels, with , height
data = np.zeros([int(t) for t in (min(MAX_FRAMES, video.get(7)), 3, video.get(3), video.get(4))])
i = 0
while video.isOpened() and i < MAX_FRAMES:
    suc, frame = video.read()
    if not suc:
        break
    n = np.sum(frame, axis=2)

    n = np.repeat(n[:,:,np.newaxis], 3, axis=2)
    d = frame / n

    data[i] = d.T

    i += 1

video.release()
print('video loaded')
data = data.astype(np.float32)

w_ = np.asarray([TARGET_COLOUR])
w_ = np.expand_dims(w_, axis=-1)
w_ = np.expand_dims(w_, axis=-1)

w_ = torch.asarray(w_.astype(np.float32))

conv = nn.Conv2d(3, 1, 1)
w_ = nn.Parameter(w_)
conv.weight = w_
conv.bias = nn.Parameter(torch.zeros((1,)))
conv = conv.to(device)
print('conv created')

thresh = Threshold(0.4)
thresh = thresh.to(device)
print('thresh created')

x = torch.from_numpy(data)
x = x.to(device)
print('x moved to gpu')

y = conv(x)
y = thresh(y)
print('ai run')

y = y.detach().cpu().numpy()
fourcc = cv.VideoWriter_fourcc(*'DIVX')
shape_ = y.shape[2:]
print('writing')
print(shape_)
writer = cv.VideoWriter('002.mov', fourcc, 25.0, shape_, isColor=False)
for d in y:
    t = (d * 255).astype(np.uint8).T
    writer.write(t)
writer.release()
print('done')

