from dataset import CARANA_DIR
from sklearn.model_selection import KFold
import cv2
import os
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F
import glob
import torch
from random import shuffle


def calculate_weight(label):
    a = F.avg_pool2d(label, kernel_size=11, padding=5, stride=1)
    ind = a.ge(0.01) * a.le(0.99)
    ind = ind.float()
    weights = Variable(torch.ones(a.size()).cuda())
    w0 = weights.sum()
    weights = weights + ind*2
    w1 = weights.sum()
    weights = weights/w1*w0
    return weights


def pred(dataloader, net, upsample=True, verbose=False):
    net.eval()
    total_size, H, W = len(dataloader.dataset.img_names), dataloader.dataset.out_h, dataloader.dataset.out_w
    pred_labels = np.empty((total_size, 1280, 1918), dtype=np.uint8) if upsample else np.empty((total_size, H, W))
    # predictions = np.empty((total_size, H, W))
    prev = 0
    for idx, (img, _) in enumerate(dataloader):
        batch_size = img.size(0)
        img = Variable(img.cuda(), volatile=True)
        scores, logits = net(img)
        if upsample:
            logits = F.upsample(logits, (1280, 1918), mode='bilinear')
        logits = logits.data.cpu().numpy()
        l = logits > 0.5
        l = np.squeeze(l)
        pred_labels[prev: prev+batch_size] = l
        prev = prev + batch_size
        if verbose:
            print('\r Progress: %.2f' % (prev/total_size), flush=True, end='')
    return pred_labels# , # predictions


def evaluate(dataloader, net, criterion):
    net.eval()
    total_size = len(dataloader.dataset.img_names)
    avg_loss = 0.0
    for img, label in dataloader:
        batch_size = img.size(0)
        img = Variable(img.cuda(), volatile=True)
        label = label.float()
        label = Variable(label.cuda(), volatile=True)
        out, logits = net(img)
        # print(logtis.data, label.data)
        # log_logits = F.log_softmax(logtis)
        # print(log_logits)
        loss = criterion(out, label, calculate_weight(label))
        avg_loss = loss.data[0]*batch_size + avg_loss
    return avg_loss/total_size


def dice_coeff(preds, targets):
    """
    preds and targets are two torch tensors with size N*H*W
    """
    targets = (targets > 0.5).astype(int)
    intersection = np.sum((preds * targets).astype(int))
    im_sum = preds.sum() + targets.sum()
    return (2. * intersection) / im_sum


def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


def save_mask(mask_imgs, model_name, names):
    # mask_imgs.astype(np.uint8)
    save_dir = os.path.join(CARANA_DIR, model_name)
    print('Save images')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for name, mask_img in zip(names, mask_imgs):
        # mask_img = cv2.cvtColor(mask_img, cv2.)
        cv2.imwrite(os.path.join(save_dir, '{}.png'.format(name)), mask_img)


def split(num=None):
    path = os.path.join(CARANA_DIR, 'split')
    img_names = glob.glob(CARANA_DIR+'/train/train/*.jpg')
    shuffle(img_names)
    if not os.path.exists(path):
        os.makedirs(path)
    if num is not None:
        with open(os.path.join(path, 'train-{}'.format(num)), 'w') as f:
            for i in range(num):
                f.write(str(img_names[i]).split('/')[-1][:-4])
                f.write('\n')
        if 5088 - num != 0:
            with open(os.path.join(path, 'valid-{}'.format(5088 - num)), 'w') as f:
                for i in range(5088-num):
                    f.write(str(img_names[i]).split('/')[-1][:-4])
                    f.write('\n')
    else:
        # read image names

        print(img_names)
        # split
        kfold = KFold(n_splits=5)
        fake_x = np.random.randn(5088, 1)
        for index, (train_index, test_index) in enumerate(kfold.split(fake_x)):
            with open(os.path.join(path, 'train-{}'.format(index)), 'w') as f:
                for t_idx in train_index:
                    f.write(str(img_names[t_idx]).split('/')[-1][:-4])
                    f.write('\n')
            with open(os.path.join(path, 'valid-{}'.format(index)), 'w') as f:
                for v_idx in test_index:
                    f.write(str(img_names[v_idx]).split('/')[-1][:-4])
                    f.write('\n')

def gta_split():
    """read train-4788 and add gta image names to this file"""
    gta_img_names = [name.split('/')[-1][:-4] for name in glob.glob(CARANA_DIR + '/train/gta_train/*.jpg')]
    shuffle(gta_img_names)
    with open(CARANA_DIR + '/split/train-4788') as f:
        content = f.readlines()
    orig_img_names = [line.strip('\n') for line in content]
    img_names = orig_img_names + gta_img_names

    with open(CARANA_DIR + '/split/train-4788-gta', 'w') as f:
        for img_name in img_names:
            f.write(img_name + '\n')
            print(img_name)
    # print(orig_img_names)


class Logger(object):
    def __init__(self, name):
        """Logger records training loss/acc per print_it, validation loss/acc per epoch, and num_mins per epoch"""
        self.name = name
        self.save_path = name
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []
        self.time = []
        self.file = open(os.path.join('logger', self.name+'.txt'), 'w')

    def log(self, train_acc, val_acc, train_loss, val_loss, time):
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.time.append(time)

    def write(self, msg, do_print=True):
        if do_print:
            print(msg)
        self.file.write(msg)
        self.file.flush()

    def save(self):
        with open(os.path.join('logger', self.name+'.txt'), 'w') as f:
            f.write('EPOCH||Train Acc||Val Acc||Train loss||Val loss||Time\n')
            for idx, (ta, va, tl, vl, t) in \
                    enumerate(zip(self.train_acc, self.val_acc, self.train_loss, self.val_loss, self.time)):
                f.write(' %s || %.5f || %.5f || %.4f || %.5f || %.4f  \n' % (idx, ta, va, tl, vl, t))


if __name__ == '__main__':
   #  from scipy.misc import imread
   #  import cv2
   #  from matplotlib import pyplot as plt
   #
   #  # split(4788)
   #  imgs = sorted(glob.glob(CARANA_DIR+'/ensemble-1/*.png'))
   #  imgs_2 = sorted(glob.glob(CARANA_DIR+'/refinenetv4_resnet34_1280*1280_hq/*.png'))
   # #  img_2 = glob.glob(CARANA_DIR+'/unet1024_5000_1/*.png')
   #  orig = sorted(glob.glob(CARANA_DIR+'/test_hq/*.jpg'))
   #
   #  for img, img_2, img_ in zip(imgs, imgs_2, orig):
   #      plt.figure(figsize=(30,20))
   #      plt.subplot(1, 2, 1)
   #      plt.title('ensemble-1')
   #      mask = (cv2.resize(imread(img), (1918, 1280)))
   #      img_ = (cv2.resize(imread(img_), (1918, 1280)))
   #      mask = cv2.resize((mask).astype(np.uint8),  (1918, 1280))
   #      mask = np.ma.masked_where(mask==0, mask)
   #      plt.gca().axis('off')
   #      plt.imshow(img_)
   #      plt.imshow(mask, 'jet', alpha=0.6)
   #
   #      plt.subplot(1, 2, 2)
   #      plt.title('1024*1824')
   #      mask = (cv2.resize(imread(img_2), (1918, 1280)))
   #      mask = cv2.resize((mask).astype(np.uint8),  (1918, 1280))
   #      mask = np.ma.masked_where(mask==0, mask)
   #      plt.imshow(img_)
   #      plt.imshow(mask, 'jet', alpha=0.6)
   #      plt.gca().axis('off')
   #      plt.show()
   gta_split()
   #      orig_ = (cv2.resize(imread(orig_), (960, 640)))
   #      cv2.imshow('f', img*100)
   #      cv2.imshow('2', img_*100)
   #      cv2.imshow('o', orig_)
   #      cv2.waitKey()
