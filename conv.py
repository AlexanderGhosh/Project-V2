import matplotlib.pyplot as plot
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video, write_video, read_video_timestamps
from torch import nn
import numpy as np
import cv2 as cv
from Timer import Timer


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

    def forward(self, x):
        x_ = torch.swapaxes(x, 1, -1) * 255.0
        x_ = x_.repeat(1, 1, 1, 3)
        return x_


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n = torch.sum(x, dim=1)[:, None, :, :]
        t = torch.div(x, n)
        return t


timer = Timer('Main Timer')
timer.start()
MAX_FRAMES = 100
TARGET_COLOUR = [0.0, 0.0, 1.0]

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
    w_ = np.asarray([TARGET_COLOUR])
    w_ = np.expand_dims(w_, axis=-1)
    w_ = np.expand_dims(w_, axis=-1)

    w_ = torch.asarray(w_).to(torch.float)
    w_ = nn.Parameter(w_)

with Timer('Model creation'):
    conv = nn.Conv2d(3, 1, 1)
    conv.weight = w_
    conv.bias = nn.Parameter(torch.zeros((1,)))

    thresh = Threshold(0.4)

    model = nn.Sequential(Normalize(), conv, thresh, ToColour())
    model.to(device)


with Timer('Run model'):
    d_ = VideoDataset('./dataset/3.mp4', MAX_FRAMES)
    video_loader = DataLoader(d_, batch_size=1, shuffle=False)
    result = torch.ones(d_.shape)
    i = 0
    for d, _ in video_loader:
        x = d.to(device)
        y = model(x)
        y = y.detach().cpu()
        result[i] = torch.transpose(y, 1, 2)
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
