from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Normalize, Lambda
import glob
import numpy as np
from PIL import Image
import cv2
import torch
import random
import time


CARANA_DIR = '/media/jxu7/BACK-UP/Data/carvana'
mean = [
    [0.68581734522164306, 0.69262389716232575, 0.69997227996210665],
    [0.68183082412592577, 0.68885076944875112, 0.6958291753151572],
    [0.68495254107142345, 0.69148648761856302, 0.69879201541420988],
    [0.68118192009717171, 0.68849969377796516, 0.69672117430285363],
    [0.68590284269603696, 0.69285547294676564, 0.69971214283887229],
    [0.68393705585379416, 0.690863245943936, 0.69820535867719524], # all
    []  # 5000
]
std = [
    [0.2451555280892101, 0.24848201503013956, 0.24391495327711973],
    [0.24551160069417943, 0.24862941368977742, 0.24465972522212173],
    [0.24367499103188969, 0.24663030373528805, 0.24287543973730677],
    [0.24534487485378531, 0.24847334712403824, 0.24370864369316059],
    [0.24506112008620376, 0.24816712564882465, 0.24457061619821843],
    [0.24495887822375081, 0.24808445495662299, 0.24395232561506316], # all
    []  # 5000
]


class CarvanaDataSet(Dataset):
    def __init__(self, split, H=256, W=256, out_h=1024, out_w=1024, transform=None,
                 test=False, preload=False, mean=None, std=None, start=None, end=None):
        # TODO: add direction information????
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
        print('[!]Loading!' + CARANA_DIR + '/' + split) if split is not None else print('Testing')

        # load sample index index
        if split is not None:
            with open(CARANA_DIR+'/split/'+split) as f:
                self.img_names = f.readlines()
            self.img_names = [name.strip('\n') for name in self.img_names]


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
            if start is not None and end is not None:
                self.img_names = glob.glob(CARANA_DIR+'/test/*.jpg')[start:end]
        print('Number of samples {}'.format(len(self.img_names)))
        print('Total loading time %.2f' % (time.time() - t))
        print('Done Loading!')

    def mean_std(self):
        mean = []
        std = []
        for i in range(0, 3):
            mean.append(np.mean(self.imgs[:, :, :, i]/255.))
            std.append(np.std(self.imgs[:, :, :, i]/255.))
        return mean, std

    def __getitem__(self, index):
        if self.test:
            img = cv2.resize(cv2.imread(self.img_names[index]), (self.W, self.H))
            img = img/255.
            if self.transform is not None:
                img = self.transform(img)
            img = toTensor(img)
            if self.normalize is not None:
                img = self.normalize(img)
            return img, 0
        else:
            if self.preload:
                img = self.imgs[index]
                label = self.labels[index]
            else:
                img_name = CARANA_DIR+'/train/'.format(self.img_names[index])
                img = cv2.imread(img_name+'train/{}.jpg'.format(self.img_names[index]))
                label = np.array(Image.open(img_name+'train_masks/{}_mask.gif'.format(self.img_names[index])))
                label = np.array(label)
            if self.transform is not None:
                img, label = self.transform((img, label))
            img = img/255.
            img, label = cv2.resize(img, (self.W, self.H)), cv2.resize(label, (self.out_w, self.out_h))
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


class RandomHueSaturationValue(object):
    def __init__(self, hue_shift_limit=(-100, 100), sat_shift_limit=(-155, 155), val_shift_limit=(-155, 155)):
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit

    def __call__(self, data):
        image, l = data
        if random.random() < 0.5:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(image)
            hue_shift = np.random.uniform(self.hue_shift_limit[0], self.hue_shift_limit[1])
            h = cv2.add(h, hue_shift)
            sat_shift = np.random.uniform(self.sat_shift_limit[0], self.sat_shift_limit[1])
            s = cv2.add(s, sat_shift)
            val_shift = np.random.uniform(self.val_shift_limit[0], self.val_shift_limit[1])
            v = cv2.add(v, val_shift)
            image = cv2.merge((h, s, v))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image, l


class RandomCrop(object):
    def __init__(self, size=(1100, 1500)):
        self.size = size

    def __call__(self, data):
        img, l = data
        u = random.random()
        if u < 0.5:
            h, w, c = img.shape
            ht, wt = self.size

            x, y = np.random.randint(0, w-wt), np.random.randint(0, h-ht)
            img = img[x:ht+x, y:wt+y]
            l = l[x:ht+x, y:+wt+y]
        return img, l


def get_valid_dataloader(batch_size, split, H=512, W=512, out_h=1280, out_w=1918, num_works=0, mean=mean, std=std):
    return DataLoader(batch_size=batch_size, num_workers=num_works,
        dataset=CarvanaDataSet(
            split, test=False, H=H, W=W, preload=True, out_h=out_h, out_w=out_w,
            transform=None, std=std, mean=mean
        )


    )


def get_train_dataloader(split, mean, std, H=512, W=512, out_h=1280, out_w=1918, batch_size=64, num_works=6):
    return DataLoader(batch_size=batch_size, shuffle=True, num_workers=num_works,
                      dataset=CarvanaDataSet(split, preload=False, H=H, W=W, out_w=out_w, out_h=out_h,
                                             test=False, mean=mean, std=std,
                                             transform=Compose([# VerticalFlip(),
                                                                RandomCrop(),
                                                                HorizontalFlip()])))


def get_test_dataloader(std, mean, H=512, W=512, out_h=1280, out_w=1918, batch_size=64, start=None, end=None):
    return DataLoader(batch_size=batch_size, num_workers=4,
                      dataset=CarvanaDataSet(start=start, end=end, split=None, H=H, W=W, std=std, mean=mean, test=True,
                                            out_h=out_h, out_w=out_w))


if __name__ == '__main__':
    # loader = get_valid_dataloader(1, H=640, W=960, preload=True, num_works=3)
    # for data in loader:
    #     i, l = data
    #     cv2.imshow('f', i.numpy()[0].transpose(1, 2, 0))
    #     cv2.imshow('l', l.numpy()[0]*100)
    #     print(l[l==1].sum())
    #     cv2.waitKey()
    # calculate mean and std for each fold
    # for i in range(5):
    #     dataset = CarvanaDataSet('train-{}'.format(i), preload=True, test=False)
    #     mean, std = dataset.mean_std()
    #     print('Fold {}--mean: {}, std: {}'.format(i, mean, std))
     #    del dataset
    dataset = CarvanaDataSet('train-5000', H=640, W=960, out_h=1280, out_w=1918, preload=True, test=False)
    print(dataset.mean_std())
