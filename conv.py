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
            img = img.permute(1, 2, 0)
            self.shape = self.len, *img.shape

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        file = f'{self.path}/{item + 1:05d}.jpg'
        return read_image(file).permute(1, 2, 0), item


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
        x = x * 255.0
        x = x.repeat(1, 1, 1, 3)
        return x


class Similar(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.loss = nn.MSELoss(reduction='none')
        self.target = torch.asarray(target).to(torch.float).to(device)

    def forward(self, x):
        t = self.loss(x, self.target)
        t = torch.sum(t, dim=-1)[:, :, :, None]
        t = torch.div(t, 3.0 ** 0.5)
        return 1.0 - t


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        t = torch.div(x, 255.0)
        return t


timer = Timer('Main Timer')
timer.start()
MAX_FRAMES = 100
TARGET_COLOUR = [0, 0, 1]

device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('cuda')
else:
    device = torch.device('cpu')
    print('cpu')


with Timer('Model creation'):
    model = nn.Sequential(Normalize(), Similar(TARGET_COLOUR), Threshold(0.7), ToColour())
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
