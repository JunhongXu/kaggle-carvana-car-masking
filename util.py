from dataset import CARANA_DIR
from scipy.misc import imread
import cv2
import os
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F


def pred(dataloader, net):
    net.eval()
    total_size, H, W = len(dataloader.dataset.img_names), dataloader.dataset.H, dataloader.dataset.W
    pred_labels = np.empty((total_size, H, W), dtype=np.uint8)
    prev = 0
    for idx, (img, _) in enumerate(dataloader):
        batch_size = img.size(0)
        # print(img.numpy())
        img = Variable(img.cuda(), volatile=True)
        _logits, _log_logits = net(img)
        # print(_logits)
        l = _logits.data.cpu().numpy()
        l = np.argmax(l, axis=1)

        pred_labels[prev: prev+batch_size] = l
        prev = prev + batch_size
        print('Batch index', idx)
    return pred_labels


def evaluate(dataloader, net, criterion):
    net.eval()
    total_size = len(dataloader.dataset.img_names)
    avg_loss = 0.0
    for img, label in dataloader:
        batch_size = img.size(0)
        img = Variable(img.cuda(), volatile=True)
        label = label.long()
        label = Variable(label.cuda(), volatile=True)
        logtis, log_logits = net(img)
        # print(logtis.data, label.data)
        # log_logits = F.log_softmax(logtis)
        # print(log_logits)
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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for name, mask_img in zip(names, mask_imgs):
        # mask_img = cv2.cvtColor(mask_img, cv2.)
        cv2.imwrite(os.path.join(save_dir, '{}.png'.format(name)), mask_img)


if __name__ == '__main__':
    import glob
    from scipy.misc import imread
    imgnames = glob.glob(CARANA_DIR+"/unet-v2/*.png")
    testimnames = glob.glob(CARANA_DIR+'/test/*.jpg')
    for testim, name in zip(testimnames, imgnames):
        img = imread(name, 'L')
        img = cv2.resize(img, (384, 256))

        test = imread(testim)
        test = cv2.resize(test, (384, 256))
    # save_mask(img, 'unte', ['1', '2'])

   #  for img in glob.glob('unte/*.png'):

        cv2.imshow('f', img)
        cv2.imshow('co', test)
        cv2.waitKey()
