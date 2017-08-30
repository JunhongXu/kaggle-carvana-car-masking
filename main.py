import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.optim import Adam, SGD
import glob
from dataset import get_valid_dataloader, get_train_dataloader, get_test_dataloader, CARANA_DIR, HorizontalFlip, \
    mean, std
from unet import UNet512, UNetV2, UNetV3
from myunet import UNet_double_1024_5, UNet_1024_5, BCELoss2d, SoftIoULoss, SoftDiceLoss
from util import pred, evaluate, dice_coeff, run_length_encode, save_mask, calculate_weight
import cv2
from scipy.misc import imread
import pandas as pd
import numpy as np
import time
from util import Logger
from math import ceil
from matplotlib import pyplot as plt

EPOCH = 70
START_EPOCH = 51
in_h = 1152
in_w = 1152
out_w = 1152
out_h = 1152
print_it = 20
interval = 30000
NUM = 100064
USE_WEIGHTING = True
model_name = 'UNET1152_1152SOFIoU'
BATCH = 4
DEBUG = False

test_aug_dim = [(1152, 1152)]





def lr_scheduler(optimizer, epoch):
    if 0 <= epoch <= 20:
        lr = 0.005
    elif 20 < epoch<= 40:
        lr = 0.001
    elif 30 < epoch <= 50:
        lr = 0.001
    else:
        lr = 0.0007
    for param in optimizer.param_groups:
        param['lr'] = lr


def train(net):
    optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)  ###0.0005
    bce2d = BCELoss2d()
    # softdice = SoftDiceLoss()
    softiou = SoftIoULoss()
    if torch.cuda.is_available():
        net.cuda()
    logger = Logger(model_name)
    if START_EPOCH > 0:
        net.load_state_dict(torch.load('models/'+model_name+'.pth'))
        print('------------Resume training %s from %s---------------' %(model_name, START_EPOCH))
    # train
    print('EPOCH || BCE loss || Avg BCE loss || Train Acc || Val loss || Val Acc || Time\n')
    best_val_loss = 0.0
    for e in range(START_EPOCH, EPOCH):
        # iterate over batches
        lr_scheduler(optimizer, e)
        moving_bce_loss = 0.0

        if e >20:
            # reduce augmentation
            train_loader.dataset.transforms = HorizontalFlip()

        num = 0
        total = len(train_loader.dataset.img_names)
        tic = time.time()
        for idx, (img, label) in enumerate(train_loader):
            net.train()
            num += img.size(0)
            img = Variable(img.cuda()) if torch.cuda.is_available() else Variable(img)
            # label = label.long()
            label = Variable(label.cuda()) if torch.cuda.is_available() else Variable(label)
            out, logits = net(img)
            # out = out.Long()
            weight = None if not USE_WEIGHTING else calculate_weight(label)
            bce_loss = bce2d(out, label, weight=weight)
            loss = bce_loss + softiou(out, label)
            moving_bce_loss += bce_loss.data[0]

            logits = logits.data.cpu().numpy() > 0.5
            train_acc = dice_coeff(np.squeeze(logits), label.data.cpu().numpy())
            # optimizer.zero_grad()
            # do backward pass
            # loss.backward()
            # update
            # optimizer.step()

            if idx == 0:
                optimizer.zero_grad()
            loss.backward()
            if idx % 5 == 0:
                optimizer.step()
                optimizer.zero_grad()

            if idx % print_it == 0:
                smooth_loss = moving_bce_loss/(idx+1)
                tac = time.time()
                print('\r %.3f || %.5f || %.5f || %.4f || ... || ... || % .2f'
                      % (num/total, bce_loss.data[0], smooth_loss, train_acc, (tac-tic)/60),
                      flush=True, end='')

        if e % 1 == 0:
            # validate
            smooth_loss = moving_bce_loss/(idx+1)
            pred_labels, _ = pred(valid_loader, net)
            valid_loss = evaluate(valid_loader, net, bce2d)
            # print(pred_labels)
            dice = dice_coeff(preds=pred_labels, targets=valid_loader.dataset.labels)
            tac = time.time()
            print('\r %s || %.5f || %.5f || %.4f || %.5f || %.4f || %.2f'
                  % (e, bce_loss.data[0], smooth_loss, train_acc, valid_loss, dice, (tac-tic)/60),
                  flush=True, end='')
            print('\n')
            logger.log(train_acc, dice, time=(tac-tic)/60, train_loss=bce_loss.data[0], val_loss=valid_loss)
            if best_val_loss < dice:
                torch.save(net.state_dict(), 'models/'+model_name+'.pth')
                best_val_loss = dice
    logger.save()


def test(net):
    """
    save the predicted mask image to .jpg file
    save the predicted mask prediction to submission file using rle_encode
    """
    # test
    upsampler = nn.Upsample(size=(out_h, out_w), mode='bilinear')
    net.load_state_dict(torch.load('models/'+model_name+'.pth'))
    if torch.cuda.is_available():
        net.cuda()

    times = ceil(NUM/interval)

    for t in range(0, int(times)):
        s = t * interval
        e = (t+1) * interval
        if t == (times -1):
            e = NUM
        # total_preds = np.zeros((e-s, out_h, out_w))
        # for (_in_h, _in_w) in test_aug_dim:
        test_loader = get_test_dataloader(batch_size=20, H=in_h, W=in_w, start=s, end=e, out_h=out_h, out_w=out_w,
                                              mean=None, std=None)
        pred_labels = pred(test_loader, net, verbose=not DEBUG, upsample=upsampler)
        # total_preds = predictions + total_preds
          #  del predictions
          #  del pred_labels
        # total_preds = total_preds >= ceil(len(test_aug_dim)/2)
        # total_preds = expit(total_preds)
        # print(total_preds.shape)
        # total_preds = total_preds + predictions
        # total_preds = total_preds / len(test_aug_dim)
        if DEBUG:
            mask = cv2.resize((pred_labels[0]).astype(np.uint8),  (1918, 1280))
            mask = np.ma.masked_where(mask==0, mask)
            plt.imshow(cv2.resize(cv2.imread(test_loader.dataset.img_names[0]), (1918, 1280)))
            plt.imshow(mask, 'jet', alpha=0.6)
            # cv2.imshow('orig', cv2.resize(cv2.imread(test_loader.dataset.img_names[0]), (1918, 1280)))
            # cv2.imshow('test', cv2.resize((total_preds[0]).astype(np.uint8),  (1918, 1280))*100)
            # cv2.waitKey()
            plt.show()
        names = glob.glob(CARANA_DIR+'/test/*.jpg')[s:e]
        names = [name.split('/')[-1][:-4] for name in names]
        # save mask
        save_mask(mask_imgs=pred_labels, model_name=model_name, names=names)
        del pred_labels




def do_submisssion():
    mask_names = glob.glob(CARANA_DIR+'/'+model_name+'/*.png')
    names = []
    rle = []
    # df = pd.DataFrame({'img'})
    for index, test_name in enumerate(mask_names):
        name = test_name.split('/')[-1][:-4]
        if index % 1000 ==0:
            print(name+'.jpg', index)
        names.append(name+'.jpg')
        mask = imread(test_name)

        rle.append(run_length_encode(cv2.resize(mask, (1918, 1280))))
    df = pd.DataFrame({'img': names, 'rle_mask': rle})
    df.to_csv(CARANA_DIR+'/'+model_name+'.csv.gz', index=False, compression='gzip')


if __name__ == '__main__':
    net = UNet_1024_5((3, in_h, in_w), 1)
    net = nn.DataParallel(net)
    # net.load_state_dict(torch.load('models/unet1024_5000.pth'))
    # from scipy.misc import imshow
    # valid_loader, train_loader = get_valid_dataloader(split='valid-88', batch_size=6, H=in_h, W=in_w, out_h=out_h,
    #                                                  out_w=out_w, mean=None, std=None), \
    #                             get_train_dataloader(split='train-5000', H=in_h, W=in_w, batch_size=BATCH, num_works=6,
    #                                                  out_h=out_h, out_w=out_w, mean=None, std=None)
    # train(net)
    # valid_loader = get_valid_dataloader(64)
    # if torch.cuda.is_available():
    #    net.cuda()
    # net = nn.DataParallel(net)
    # net.load_state_dict(torch.load('models/unet-v1-640*960.pth'))


    # print(evaluate(valid_loader, net, nn.NLLLoss2d()))
    # pred_labels = pred(valid_loader, net)

    # for l in pred_labels:
        # print(l.sum())
        # print(l)
        # imshow( l)
    # names = glob.glob(CARANA_DIR+'/test/*.jpg')
    # names = [name.split('/')[-1][:-4] for name in names]
    # save mask

    # save_mask(mask_imgs=pred_labels, model_name='unet', names=names)

    test(net)
    do_submisssion()
