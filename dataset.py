from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Normalize, Lambda
import glob
import numpy as np
from PIL import Image
import cv2
import torch
import random
import time
from inspect import signature


CARANA_DIR = '/media/jxu7/BACK-UP/Data/carvana'
mean = [0.68404490794406181, 0.69086280353012897, 0.69792341323303619]
std = [0.24479961336692371, 0.24790616166162652, 0.24398260796692428]


class CarvanaDataSet(Dataset):
    def __init__(self, split, H=256, W=256, out_h=1024, out_w=1024, transform=None,
                 test=False, preload=False, mean=None, std=None):
        super(CarvanaDataSet, self).__init__()
        self.H, self.W = H, W
        self.out_h = out_h
        self.out_w = out_w
        self.mean = mean
        self.std = std
        self.transform = transform
        self.test = test
        self.preload = preload
        self.img_names = []
        self.normalize = Normalize(mean=mean, std=std) if mean is not None and std is not None else None
        t = time.time()
        print('[!]Loading!' + CARANA_DIR + '/' + split)

        # load sample index index
        with open(CARANA_DIR+'/split/'+split) as f:
            self.img_names = f.readlines()
        self.img_names = [name.strip('\n') for name in self.img_names]
        print('Number of samples %s'.format(len(self.img_names)))

        if preload:
            self.imgs = np.empty((len(self.img_names), H, W, 3))
            for i, name in enumerate(self.img_names):
                img = cv2.imread(CARANA_DIR+'/train/train/{}.jpg'.format(name))
                img = cv2.resize(img, (W, H))
                self.imgs[i] = img

            if not test:
                self.labels = np.empty((len(self.img_names), out_h, out_w))
                for i, name in enumerate(self.img_names):
                    l = Image.open(CARANA_DIR+'/train/train_masks/{}_mask.gif'.format(name)).resize((out_w, out_h))
                    self.labels[i] = l

        if test:
            self.img_names = glob.glob(CARANA_DIR+'/test/*.jpg')

        print('Total loading time %.2f' % (time.time() - t))
        print('Done Loading!')

    def mean_std(self):
        mean = []
        std = []
        for i in range(0, 3):
            mean.append(np.mean(self.imgs[:, :, :, i]))
            std.append(np.std(self.imgs[:, :, :, i]))
        return mean, std

    def __getitem__(self, index):
        if self.test:
            img = cv2.resize(cv2.imread(self.img_names[index]), (self.W, self.H))
            if self.transform is not None:
                img = self.transform(img/255.)
            return img, 0
        else:
            if self.preload:
                img = self.imgs[index]
                label = self.labels[index]
            else:
                img_name = CARANA_DIR+'/train/'.format(self.img_names[index])
                img = cv2.resize(cv2.imread(img_name+'train/{}.jpg'.format(self.img_names[index])), (self.W, self.H))
                label = Image.open(img_name+'train_masks/{}_mask.gif'.format(self.img_names[index])).\
                    resize((self.out_w, self.out_h))
                label = np.array(label)

            img = img/255.
            if self.transform is not None:
                # image transform on both mask and image
                img, label = self.transform((img, label))
            img = toTensor(img)
            label = toTensor(label)
            if self.normalize is not None:
                img = self.normalize(img)
            return img, label

    def __len__(self):
        return len(self.img_names)


def toTensor(img):
    if len(img.shape) < 3:
        return torch.from_numpy(img).float()
    else:
        img = img.transpose((2, 0, 1)).astype(np.float32)
        tensor = torch.from_numpy(img).float()
        return tensor


class HorizontalFlip(object):
    def __call__(self, data):
        u = random.random()
        img, l = data
        if u < 0.5:
            img = cv2.flip(img, 0)
            l = cv2.flip(l, 0)
        return img, l


class VerticalFlip(object):
    def __call__(self, data):
        u = random.random()
        img, l = data
        if u < 0.5:
            img = cv2.flip(img, 1)
            l = cv2.flip(l, 1)
        return img, l


def get_valid_dataloader(batch_size, split, H=512, W=512, preload=False, num_works=0):
    return DataLoader(batch_size=batch_size, num_workers=num_works,
        dataset=CarvanaDataSet(
            split, test=False, H=H, W=W, preload=True,
            transform=None
        )


    )


def get_train_dataloader(split, H=512, W=512, batch_size=64, preload=False, num_works=0):
    return DataLoader(batch_size=batch_size, shuffle=True, num_workers=num_works,
                      dataset=CarvanaDataSet(split, preload=False, H=H, W=W, test=False,
                                             transform=Compose([VerticalFlip(), HorizontalFlip()])))


def get_test_dataloader(H=512, W=512, batch_size=64):
    return DataLoader(batch_size=batch_size, num_workers=4,
                      dataset=CarvanaDataSet(H=H, W=W, transform=Compose([Lambda(lambda x: toTensor(x)),
                                                                Normalize(mean=mean, std=std)]), test=True))


if __name__ == '__main__':
    loader = get_valid_dataloader(1, H=640, W=960, preload=True, num_works=3)
    for data in loader:
        i, l = data
        cv2.imshow('f', i.numpy()[0].transpose(1, 2, 0))
        cv2.imshow('l', l.numpy()[0]*100)
        print(l[l==1].sum())
        cv2.waitKey()
