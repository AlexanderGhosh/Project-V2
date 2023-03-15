import matplotlib.pyplot as plot
import torch
import torchvision.io
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video, write_video, read_video_timestamps, read_image, ImageReadMode
from torch import nn
import numpy as np
import cv2 as cv
from Timer import Timer
import os

INV_COLOUR = 255.0 ** -1.0


class VideoDataset2(Dataset):
    def __init__(self, path):
        self.path = path
        self.len = 0
        self.shape = (0, 0, 0)
        if os.path.isdir(path):
            self.len = len(os.listdir(path))
            img = read_image(f'{self.path}/00001.jpg')
            self.shape = self.len, *img.shape

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        file = f'{self.path}/{item + 1:05d}.jpg'
        return read_image(file), item


class VideoDataset(Dataset):
    def __init__(self, file, max):
        self.loc = file
        meta = read_video_timestamps(file)
        self.frame_count = min(max, len(meta[0]))
        self.fps = meta[1]
        data = read_video(self.loc, 0, 0)[0]
        self.shape = (self.frame_count, *data.shape[1:])

    def __len__(self):
        return self.frame_count

    # item, label
    def __getitem__(self, item):
        data = read_video(self.loc, item, item, output_format='TCHW')
        return data[0][0], 0


class Threshold(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        return torch.abs(torch.ceil(x - self.threshold - 0.001))


class ToColour(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()

    def forward(self, x):
        # x = self.activation(x[:, 0, :, :] + x[:, 1, :, :])
        x = x[:, 0, :, :]
        x = x * 255.0
        x = x.repeat(1, 3, 1, 1)[0]
        return x


class Similar(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target

    def forward(self, x):
        return torch.abs(1.0 - (self.target - x))


def norm(x):
    i = x[0] + 256.0 * (x[1] + x[2] * 256.0)
    i *= INV_COLOUR
    i *= INV_COLOUR
    i *= INV_COLOUR
    return i


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        i = x[:, 0, :, :] + 256.0 * (x[:, 1, :, :] + x[:, 2, :, :] * 256.0)
        i *= INV_COLOUR
        i *= INV_COLOUR
        i *= INV_COLOUR
        i = i[:, None, :, :]
        # n = torch.sum(x, dim=1)[:, None, :, :]
        t = torch.pow(x, 0)
        t = torch.mul(t, i)[:, 0, :, :][:, None, :, :]
        return t


timer = Timer('Main Timer')
timer.start()
MAX_FRAMES = 100
TARGET_COLOUR = [0, 0, 255]

device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('cuda')
else:
    device = torch.device('cpu')
    print('cpu')

# im = cv.imread('./dataset/1.jpg')
# im = cv.cvtColor(im, cv.COLOR_BGR2RGB).astype(np.float32)
'''with Timer('Loading video into memory'):
    # video = cv.VideoCapture('./dataset/3.mp4')
    # frame count, channels, with , height
    # data = np.zeros([int(t) for t in (min(MAX_FRAMES, video.get(7)), 3, video.get(3), video.get(4))])
    data = read_video('./dataset/3.mp4', output_format='TCHW', end_pts=MAX_FRAMES-1)
    data = data[0].to(torch.float)
    '' i = 0
    while video.isOpened() and i < MAX_FRAMES:
        suc, frame = video.read()
        if not suc:
            break
        n = np.sum(frame, axis=2)

        n = np.repeat(n[:, :, np.newaxis], 3, axis=2)
        d = frame / n

        data[i] = d.T

        i += 1'''

# video.release()

'''data = data.astype(np.float32)
    x = torch.from_numpy(data)''
    x = data.to(device)
'''
with Timer('Convolution weight creation'):
    w_ = np.asarray([TARGET_COLOUR, [1, 1, 0]])
    w_ = np.expand_dims(w_, axis=-1)
    w_ = np.expand_dims(w_, axis=-1)

    w_ = torch.asarray(w_).to(torch.float)
    w_ = nn.Parameter(w_)

with Timer('Model creation'):
    conv = nn.Conv2d(3, 2, 1)
    conv.weight = w_
    conv.bias = nn.Parameter(torch.zeros(conv.bias.shape))

    thresh = Threshold(0.4)

    model = nn.Sequential(Normalize(), Similar(norm(TARGET_COLOUR)), thresh, ToColour())
    model.to(device)

with Timer('Run model'):
    d_ = VideoDataset2('../dataset/videos/class0/001')
    video_loader = DataLoader(d_, batch_size=1, shuffle=False)
    result = torch.ones(d_.shape)
    i = 0
    for d, _ in video_loader:
        x = d.to(device)
        y = model(x)
        y = y.detach().cpu()
        result[i] = y
        i += 1

with Timer('Save data'):
    result = torch.transpose(result, 1, -1)
    result = torch.transpose(result, 1, 2)
    write_video('002.mov', result, 25)
    '''shape_ = y.shape[2:]
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    writer = cv.VideoWriter('002.mov', fourcc, 25.0, shape_, isColor=False)
    for d in y:
        t = (d * 255).astype(np.uint8).T
        writer.write(t)
    writer.release()'''

print('done')
timer.stop()
