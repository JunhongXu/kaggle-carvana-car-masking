from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Normalize, Lambda
import glob
import numpy as np
from PIL import Image
import cv2
import torch


CARANA_DIR = '/media/jxu7/BACK-UP/Data/carvana'
mean = [0.68404490794406181, 0.69086280353012897, 0.69792341323303619]
std = [0.24479961336692371, 0.24790616166162652, 0.24398260796692428]


class CarvanaDataSet(Dataset):
    def __init__(self, H=256, W=int(256*1.5), valid=False, transform=None, test=False):
        super(CarvanaDataSet, self).__init__()
        self.H, self.W = H, W
        self.transform = transform
        self.test = test
        print('[!]Loading!' + CARANA_DIR)
        if not test:
            all_imgs = sorted(glob.glob(CARANA_DIR+'/train/train/*.jpg'))
            all_label_names = sorted(glob.glob(CARANA_DIR + '/train/train_masks/*gif'))
            # read images
            if valid:
                self.img_names = all_imgs[-100:]
                self.label_names = all_label_names[-100:]
                self.imgs = np.empty((len(self.img_names), H, W, 3))
            else:
                self.img_names = all_imgs[:-100]
                self.label_names = all_label_names[:-100]
                self.imgs = np.empty((len(self.img_names), H, W, 3))

            self.labels = np.empty((len(self.label_names), H, W))
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
            img = self.imgs[index]
            if self.transform is not None:
                img = self.transform(img)
            return img/255., self.labels[index]
        else:
            img = cv2.resize(cv2.imread(self.img_names[index]), (self.W, self.H))
            if self.transform is not None:
                img = self.transform(img)
            return img/255., 0

    def __len__(self):
        return len(self.imgs)


def toTensor(img):
    img = img.transpose((2, 0, 1)).astype(np.float32)
    tensor = torch.from_numpy(img).float()
    return tensor


def get_valid_dataloader(batch_size):
    return DataLoader( batch_size=batch_size,
        dataset=CarvanaDataSet(
            valid=True,
            transform=Compose(
                [Lambda(lambda x: toTensor(x)), Normalize(mean=mean, std=std)]
            )
        )


    )


def get_train_dataloader(batch_size=64):
    return DataLoader(batch_size=batch_size, dataset=CarvanaDataSet(transform=Compose([Lambda(lambda x: toTensor(x)),
                                                                Normalize(mean=mean, std=std)])))


def get_test_dataloader(batch_size=64):
    return DataLoader(batch_size=batch_size,
                      dataset=CarvanaDataSet(transform=Compose([Lambda(lambda x: toTensor(x)),
                                                                Normalize(mean=mean, std=std)]), test=True))


if __name__ == '__main__':
    loader = get_valid_dataloader(64)
    for img, l in loader:
        print(img.size())
    # print(loader.dataset.mean_std())
    cv2.imshow('frame', loader.dataset.imgs[0])
    cv2.imshow('label', loader.dataset.labels[0])
    cv2.waitKey()
