from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Normalize, Lambda
import glob
import numpy as np
from PIL import Image
import cv2
import torch
import random


CARANA_DIR = '/Users/JunhongXu/Desktop/kaggle/Carvana'
mean = [0.68404490794406181, 0.69086280353012897, 0.69792341323303619]
std = [0.24479961336692371, 0.24790616166162652, 0.24398260796692428]


class CarvanaDataSet(Dataset):
    def __init__(self, H=256, W=int(256*1.5), valid=False, transform=None, test=False, preload=False):
        super(CarvanaDataSet, self).__init__()
        self.H, self.W = H, W
        self.transform = transform
        self.test = test
        self.preload = preload
        print('[!]Loading!' + CARANA_DIR)
        if not test:
            all_imgs = sorted(glob.glob(CARANA_DIR+'/train/train/*.jpg'))
            all_label_names = sorted(glob.glob(CARANA_DIR + '/train/train_masks/*gif'))
            # read images
            if valid:
                self.img_names = all_imgs[-100:]
                self.label_names = all_label_names[-100:]
                if not self.preload:
                    self.imgs = np.empty((len(self.img_names), H, W, 3))
            else:
                self.img_names = all_imgs[:-100]
                print(len(self.img_names))
                self.label_names = all_label_names[:-100]
                if not self.preload:
                    self.imgs = np.empty((len(self.img_names), H, W, 3))

            if not self.preload:
                self.labels = np.empty((len(self.label_names), H, W))
                for idx, img_name in enumerate(self.img_names):
                    self.imgs[idx] = cv2.resize(cv2.imread(img_name), (W, H))
                for idx, label_name in enumerate(self.label_names):
                    l = Image.open(label_name).resize((W, H))
                    l = np.array(l)
                    self.labels[idx] = l
        else:
            # in test mode, read the test image on the fly
            self.img_names = glob.glob(CARANA_DIR+'/test/*.jpg')
        #
        #     self.imgs = np.zeros((len(self.img_names), H, W, 3), dtype=np.uint8)
        # for idx, img_name in enumerate(self.img_names):
        #     self.imgs[idx] = cv2.resize(cv2.imread(img_name), (W, H))
        # self.imgs.astype(np.float32)
        # self.imgs = self.imgs/255.
        print('Done Loading!')

    def mean_std(self):
        mean = []
        std = []
        for i in range(0, 3):
            mean.append(np.mean(self.imgs[:, :, :, i]))
            std.append(np.std(self.imgs[:, :, :, i]))
        return mean, std

    def __getitem__(self, index):
        if not self.test:
            if self.preload:
                img = cv2.resize(cv2.imread(self.img_names[index]), (self.W, self.H))
                label = Image.open(self.label_names[index]).resize((self.W, self.H))
                label = np.array(label)
            else:
                img = self.imgs[index]
                label = self.labels[index]
            if self.transform is not None:
                if len(self.transform.transforms) > 2:
                    for t in self.transform.transforms[:-2]:
                        img, label = t(img, label)
                    for t in self.transform.transforms[-2:]:
                        img = t(img)
            return img/255., label
        else:
            img = cv2.resize(cv2.imread(self.img_names[index]), (self.W, self.H))
            if self.transform is not None:
                img = self.transform(img)
            return img/255., 0

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
    def __call__(self, img, l):
        u = random.random()
        if u < 0.5:
            img = cv2.flip(img, 0)
            l = cv2.flip(l, 0)
        return img, l


class VerticalFlip(object):
    def __call__(self, img,l):
        u = random.random()
        if u < 0.5:
            img = cv2.flip(img, 1)
            l = cv2.flip(l, 1)
        return img, l


def get_valid_dataloader(batch_size, H=512, W=512, preload=False, num_works=0):
    return DataLoader(batch_size=batch_size, num_workers=num_works,
        dataset=CarvanaDataSet(
            valid=True,
            H=H, W=W, preload=preload,
            transform=Compose(
                [VerticalFlip(), HorizontalFlip(), Lambda(lambda x: toTensor(x)),
                                                                 Normalize(mean=mean, std=std)]
            )
        )


    )


def get_train_dataloader(H=512, W=512, batch_size=64, preload=False, num_works=0):
    return DataLoader(batch_size=batch_size, shuffle=True, num_workers=num_works,
                      dataset=CarvanaDataSet(preload=preload, H=H, W=W, transform=Compose([VerticalFlip(),
                                                                                           HorizontalFlip(),
                                                                           Lambda(lambda x: toTensor(x)),
                                                                 Normalize(mean=mean, std=std)])))


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