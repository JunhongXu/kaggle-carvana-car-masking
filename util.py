from dataset import CARANA_DIR
from scipy.misc import imread
import cv2
import os
from torch.autograd import Variable
import numpy as np


def pred(dataloader, net):
    net.eval()
    total_size, H, W, C = dataloader.dataset.imgs.shape
    logits = np.empty((total_size, 2, H, W))
    log_logtis = np.empty((total_size, 2, H, W))
    prev = 0
    for img, _ in dataloader:
        batch_size = img.size(0)
        img = Variable(img.cuda(), volatile=True)
        _logits, _log_logits = net(img)
        logits[prev: prev+batch_size] = _logits.data.cpu().numpy()
        log_logtis[prev: prev+batch_size] = _log_logits.data.cpu().numpy()
        prev = prev + batch_size

    return logits, log_logtis


def evaluate(dataloader, net, criterion):
    net.eval()
    total_size = len(dataloader.dataset.imgs)
    avg_loss = 0.0
    for img, label in dataloader:
        batch_size = img.size(0)
        img = Variable(img.cuda(), volatile=True)
        label = label.long()
        label = Variable(label.cuda(), volatile=True)
        logtis, log_logits = net(img)
        loss = criterion(log_logits, label)
        avg_loss = loss.data[0]*batch_size + avg_loss
    return avg_loss/total_size


def dice_coeff(preds, targets):
    """
    preds and targets are two torch tensors with size N*H*W
    """
    intersection = np.sum((preds * targets).astype(int))
    im_sum = preds.sum() + targets.sum()
    return (2. * intersection) / im_sum


def rle_encode(mask_image):
    """https://www.kaggle.com/stainsby/fast-tested-rle"""
    pixels = mask_image.flatten()
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def save_mask(mask_imgs, model_name, names):
    mask_imgs.astype(np.uint8)
    # save_dir = os.path.join(CARANA_DIR, model_name)
    for name, mask_img in zip(names, mask_imgs):
        # mask_img = cv2.cvtColor(mask_img, cv2.)
        cv2.imwrite(os.path.join(model_name, '{}.png'.format(name)), mask_img)


if __name__ == '__main__':
    import glob
    img = np.random.randint(0, 2, (2, 300, 300))
    save_mask(img, 'unte', ['1', '2'])

    for img in glob.glob('unte/*.png'):
        a = imread(img, 'L')
        print(a)
        cv2.imshow('f', a)
        cv2.waitKey()