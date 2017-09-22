import glob
import math
import random
import time
from scipy.misc import imread
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision.transforms import Compose, Normalize, ToTensor, RandomHorizontalFlip

from miscs.cls_labels import label2idx, idx2label

random.seed(0)

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


class CarClassificationDataset(Dataset):
    def __init__(self, split, h, w, transform):
        super(CarClassificationDataset, self).__init__()
        label_df = pd.read_csv(CARANA_DIR+'/model_labels.csv')
        self.labels = []
        self.img_names = []
        self.h, self.w = h, w
        self.transform = transform
        with open(split) as f:
            content = f.readlines()
        img_names = [line.strip('\n') for line in content]
        for idx, img_name in enumerate(img_names):
            print('\r[!]Checking %.2f' % (idx/len(img_names)), flush=True, end='')
            id = img_name.split('/')[-1][:-7]
            label = label_df.loc[label_df['id'] == id]
            # this image has a model
            if len(label) != 0:
                self.img_names.append(img_name)
                self.labels.append(label2idx[label.get_values()[-1][-1]])

    def __getitem__(self, index):
        img = cv2.resize(cv2.imread(self.img_names[index]), (self.w, self.h))
        img = Image.fromarray(img)
        label = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        # img = img / 255.
        # img = toTensor(img)
        return img, label

    def __len__(self):
        return len(self.img_names)


class PesudoLabelCarvanaDataSet(Dataset):
    """This is just for the training set"""
    def __init__(self, split='train_pesudo', H=1024, W=1024, out_h=1024, out_w=1024, transform=None):
        super(PesudoLabelCarvanaDataSet, self).__init__()
        self.H = H
        self.W = W
        self.out_h = out_h
        self.out_w = out_w
        self.transform = transform

        self.img_names = []     # combines with train and test image names
        self.mask_names = []
        with open(CARANA_DIR + '/split/' + split) as f:
            content = f.readlines()
        content = [line.strip('\n') for line in content]
        self.num_pesudo = 0
        self.num_sample = 0
        for name in content:
            self.img_names.append(CARANA_DIR + name + '.jpg')
            img_name = name.split('/')[-1]
            if 'test' in name:
                self.mask_names.append(CARANA_DIR+'/train/train_pesudo_masks/{}.tiff'.format(img_name))
                self.num_pesudo += 1
            else:
                self.num_sample += 1
                self.mask_names.append(CARANA_DIR+'/train/train_masks/{}_mask.gif'.format(img_name))
        print('[!]Number of pesudo training samples:', self.num_pesudo)
        print('[!]Number of training samples:', self.num_sample)

    def __getitem__(self, index):
        img = cv2.imread(self.img_names[index])
        mask = np.array(Image.open(self.mask_names[index]))
        if self.transform is not None:
            img, mask = self.transform((img, mask))

        img = img/255.
        img, mask = cv2.resize(img, (self.out_w, self.out_h)), cv2.resize(mask, (self.out_w, self.out_h))

        img = toTensor(img)
        mask = toTensor(mask)
        return img, mask, self.mask_names[index]

    def __len__(self):
        return self.num_sample


class CarvanaDataSet(Dataset):
    def __init__(self, split, H=256, W=256, out_h=1024, out_w=1024, transform=None, hq=True,
                 test=False, preload=False, mean=None, std=None, start=None, end=None, num=None):
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
        self.num = num
        self.normalize = Normalize(mean=mean, std=std) if mean is not None and std is not None else None
        self.use_gta = True if split is not None and 'gta' in split else False

        t = time.time()
        print('[!]Loading!' + CARANA_DIR + '/' + split) if split is not None else print('Testing')

        # load sample index index
        if split is not None:
            with open(CARANA_DIR+'/split/'+split) as f:
                self.img_names = f.readlines()
                self.img_names = [name.strip('\n') for name in self.img_names]
        if not test:
            self.load_dir = CARANA_DIR + '/train/train_hq/' if hq else CARANA_DIR + '/train/train/'
            if preload:
                self.imgs = np.empty((len(self.img_names), H, W, 3))
                for i, name in enumerate(self.img_names):
                    img = cv2.imread(self.load_dir + '{}.jpg'.format(name))
                    img = cv2.resize(img, (W, H))
                    self.imgs[i] = img

            if 'valid' in split:
                    self.labels = np.empty((len(self.img_names), out_h, out_w))
                    for i, name in enumerate(self.img_names):
                        l = Image.open(CARANA_DIR+'/train/train_masks/{}_mask.gif'.format(name)).resize((out_w, out_h))
                        self.labels[i] = l
        else:
            self.img_names = glob.glob(CARANA_DIR+'/test/*.jpg' if not hq else CARANA_DIR+'/test_hq/*.jpg')
            if start is not None and end is not None:
                self.img_names = self.img_names[start:end]
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
            return img, self.img_names[index]
        else:
            if self.preload:
                img = self.imgs[index]
                label = self.labels[index]
            else:
                # img_name = CARANA_DIR+'/train/'.format(self.img_names[index])
                img = cv2.imread(self.load_dir+'{}.jpg'.format(self.img_names[index]))
                label_name = self.img_names[index]
                # if 'png' in label_name:
                try:
                    label_mask = CARANA_DIR + '/train/train_masks/{}.png'.format(label_name)
                    label = imread(label_mask, mode='L')
                    label[label !=0] = 1
                    # print(label[label != 0])
                    # plt.imshow(label)
                    # plt.show()
                except:
                    label_mask = CARANA_DIR + '/train/train_masks/{}_mask.gif'.format(label_name)
                    label = np.array(Image.open(label_mask))
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
        return self.num if self.num is not None else len(self.img_names)


def toTensor(img):
    if len(img.shape) < 3:
        return torch.from_numpy(img).float()
    else:
        img = img.transpose((2, 0, 1)).astype(np.float32)
        tensor = torch.from_numpy(img).float()
        return tensor


class VerticalFlip(object):
    def __call__(self, data):
        u = random.random()
        img, l = data
        if u < 0.5:
            img = cv2.flip(img, 0)
            l = cv2.flip(l, 0)
        return img, l


class HorizontalFlip(object):
    def __call__(self, data, u=0.5):
        img, l = data
        if random.random() < u:
            img = cv2.flip(img, 1)
            l = cv2.flip(l, 1)
        return img, l


class RandomRotate(object):
    def __init__(self, shift_limit=(-0.062, 0.062), scale_limit=(0.91,1.21), rotate_limit=(-10, 10),
                 aspect_limit=(1, 1), ):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.aspect_limit = aspect_limit

    def __call__(self, data, borderMode=cv2.BORDER_REFLECT_101, u=0.5):

        # cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT
        img, l = data
        if random.random() < u:
            height, width, channel = img.shape

            angle = random.uniform(self.rotate_limit[0], self.rotate_limit[1])  # degree
            scale = random.uniform(self.scale_limit[0], self.scale_limit[1])
            aspect = random.uniform(self.aspect_limit[0], self.aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = round(random.uniform(self.shift_limit[0], self.shift_limit[1]) * width)
            dy = round(random.uniform(self.shift_limit[0], self.shift_limit[1]) * height)

            cc = math.cos(angle / 180 * math.pi) * (sx)
            ss = math.sin(angle / 180 * math.pi) * (sy)
            rotate_matrix = np.array([[cc, -ss], [ss, cc]])

            box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
            box1 = box0 - np.array([width / 2, height / 2])
            box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

            box0 = box0.astype(np.float32)
            box1 = box1.astype(np.float32)
            mat = cv2.getPerspectiveTransform(box0, box1)

            img = cv2.warpPerspective(img, mat, (width, height), flags=cv2.INTER_LINEAR,  borderMode=borderMode,
                                      borderValue=(0, 0, 0,))
            l = cv2.warpPerspective(l, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(0, 0, 0,))

        return img, l


class RandomHueSaturationValue(object):
    def __init__(self, hue_shift_limit=(-50, 50), sat_shift_limit=(-5, 5), val_shift_limit=(-15, 15)):
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit

    def __call__(self, data, u=0.5):
        image, l = data
        if random.random() < u:
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


class RandomTransposeColor(object):
    def __call__(self, data, u=0.5):
        img, l = data
        if random.random() < u:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, l


class RandomGray(object):
    def __call__(self, data, u=0.3):
        img, l = data
        if random.random() < u:
            coeff = np.array([.7154, .0721, .2125])
            gray_img = np.sum(img * coeff, 2)
            img[:, :, 0] = gray_img
            img[:, :, 1] = gray_img
            img[:, :, 2] = gray_img
        return img, l


class RandomCrop(object):
    def __init__(self, size=(1240, 1880)):
        self.size = size

    def __call__(self, data, u=0.5):
        img, l = data
        if random.random() < u:
            h, w, c = img.shape
            ht, wt = self.size
            x, y = np.random.randint(0, w-wt), np.random.randint(0, h-ht)
            img = img[x:ht+x, y:wt+y]
            l = l[x:ht+x, y:+wt+y]
        return img, l

##################### transform types ###############
transform1 = Compose([
    RandomCrop(),
    HorizontalFlip()
])

transform2 = HorizontalFlip()


transform3 = Compose(
    [
        HorizontalFlip(),
        # RandomTransposeColor(),
        RandomRotate(rotate_limit=(0, 0))
    ]
)

transform4 = Compose([
    HorizontalFlip(),
    RandomTransposeColor(),
    RandomHueSaturationValue(),
    RandomRotate(),
    RandomGray(),
])


def get_valid_dataloader(batch_size, split, H=512, W=512, out_h=1280, out_w=1918, preload=True, num_works=0, mean=mean, std=std):
    return DataLoader(batch_size=batch_size, num_workers=num_works,
        dataset=CarvanaDataSet(
            split, test=False, H=H, W=W, preload=preload, out_h=out_h, out_w=out_w,
            transform=None, std=std, mean=mean
        )


    )


def get_train_dataloader(split, mean, std, transforms=Compose([RandomCrop(), HorizontalFlip()]), num=4788,
                         H=512, W=512, out_h=1280, out_w=1918, batch_size=64, num_works=6, preload=False):
    return DataLoader(batch_size=batch_size, shuffle=True, num_workers=num_works,
                      dataset=CarvanaDataSet(split, preload=preload, H=H, W=W, out_w=out_w, out_h=out_h,
                                             test=False, mean=mean, std=std,
                                             transform=transforms, num=num))


def get_pesudo_train_dataloader(in_h, in_w, out_h, out_w, batch_size, num_workers=6):
    dataset = PesudoLabelCarvanaDataSet(H=in_h, W=in_w, out_h=out_h, out_w=out_w, transform=transform3)

    return DataLoader(
        batch_size=batch_size, sampler=PesudoSampler(dataset), num_workers=num_workers,
        dataset=dataset
    )


def get_test_dataloader(std, mean, H=512, W=512, out_h=1280, out_w=1918, batch_size=64, start=None, end=None,
                        load_number=None):
    return DataLoader(batch_size=batch_size, num_workers=4,
                      dataset=CarvanaDataSet(start=start, end=end, split=None, H=H, W=W, std=std,
                                             mean=mean, test=True,
                                            out_h=out_h, out_w=out_w))


def get_cls_train_dataloader(in_h=256, in_w=256, batch_size=128, num_workers=6):
    return DataLoader(
        batch_size=batch_size, num_workers=num_workers,
        dataset=CarClassificationDataset(CARANA_DIR+'/split/train-class', h=in_h, w=in_w,
                                         transform=Compose([
                                             RandomHorizontalFlip(),
                                             ToTensor()
                                         ]))
    )


def get_cls_valid_dataloader(in_h=256, in_w=256, batch_size=128, num_workers=6):
    return DataLoader(
        batch_size=batch_size, num_workers=num_workers,
        dataset=CarClassificationDataset(CARANA_DIR+'/split/valid-class', h=in_h, w=in_w,
                                         transform=Compose([
                                             ToTensor()
                                         ]))
    )


class PesudoSampler(Sampler):
    def __init__(self, data_source):
        super(PesudoSampler, self).__init__(None)
        self.num_pesudo = data_source.num_pesudo
        self.num_sample = data_source.num_sample
        self.sample_prob = 0.7
        self.pesudo_prob = 0.3
        self.data_source = data_source
        self.weights = [self.sample_prob / self.num_sample for _ in range(self.num_sample)] + \
                       [self.pesudo_prob / self.num_pesudo for _ in range(self.num_pesudo)]
        self.weights = torch.DoubleTensor(self.weights)

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_sample, True))

    def __len__(self):
        return self.num_sample


if __name__ == '__main__':
    # sampler = BatchSampler(PesudoSampler(), 64, False)
    # num_sample = 0
    # num_pesudo = 0
    # for idx in sampler:
    #     idx = np.array(idx)
    #     num_sample += np.sum((idx < 4788).astype(int))
    #     num_pesudo += np.sum((idx >= 4788).astype(int))
    #     print(idx)
    # print(num_pesudo/num_sample)
    loader =  get_train_dataloader(split='train-4788-gta', H=512, W=512, batch_size=10, num_works=6, num=None,
                                                              out_h=512, out_w=512, mean=None, std=None, transforms=transform3)
    for imgs, labels, in loader:
        for img, label in zip(imgs, labels):
            img = img.cpu().numpy()
            img = img.transpose(1, 2, 0)
            # mask = np.ma.masked_where(mask == 0, mask)
            plt.imshow(img)
           # print(idx2label[label])
            plt.show()
