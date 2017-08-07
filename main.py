import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, SGD
import glob
from dataset import get_valid_dataloader, get_train_dataloader, get_test_dataloader, CARANA_DIR
from unet import UNetV1, UNetV2, UNetV3
from util import pred, evaluate, dice_coeff, run_length_encode, save_mask
import numpy as np
import cv2
from scipy.misc import imread
import pandas as pd


EPOCH = 60
LEARNING_RATE = 5e-4
L2_DECAY = 5e-4


def lr_scheduler(optimizer, epoch):
    if 0 <= epoch <= 10:
        lr = 0.005
    elif 10 < epoch<= 35:
        lr = 0.0009
    elif 35 < epoch <= 40:
        lr = 0.0005
    else:
        lr = 0.0001
    for param in optimizer.param_groups:
        param['lr'] = lr


def train(net):
    # valid_loader = get_valid_dataloader(20)
    # optimizer = Adam(params=net.parameters(), lr=LEARNING_RATE, weight_decay=L2_DECAY)
    optimizer = SGD(params=net.parameters(), lr=LEARNING_RATE, weight_decay=L2_DECAY, momentum=0.9)
    criterion = nn.NLLLoss2d()
    if torch.cuda.is_available():
        net.cuda()
    net = nn.DataParallel(net)
    print(net)
    # train
    best_val_loss = 0.0
    for e in range(EPOCH):
        # iterate over batches
        lr_scheduler(optimizer, e)
        num = 0
        total = len(train_loader.dataset.img_names)
        for idx, (img, label) in enumerate(train_loader):
            net.train()
            num += img.size(0)
            img = Variable(img.cuda()) if torch.cuda.is_available() else Variable(img)
            label = label.long()
            label = Variable(label.cuda()) if torch.cuda.is_available() else Variable(label)
            logits, log_logits = net(img)
            loss = criterion(log_logits, label)
            # fresh gradients
            optimizer.zero_grad()
            # do backward pass
            loss.backward()
            # update
            optimizer.step()

            if idx % 10 == 0:
                print('\r {}: Training loss is'.format(num/total), loss.data[0], flush=True, end='')
        if e % 1 == 0:
            # validate
            pred_labels = pred(valid_loader, net)
            valid_loss = evaluate(valid_loader, net, criterion)
            # print(pred_labels)
            dice = dice_coeff(preds=pred_labels, targets=valid_loader.dataset.labels)
            print('\nEpoch {}: validation loss-{}, dice coeff-{}, best loss-{}'.format(e, valid_loss, dice, best_val_loss))
            if best_val_loss < dice:
                print('Save')
                torch.save(net.state_dict(), 'models/unet-v1-768*1152.pth')
                best_val_loss = dice




def test(net):
    """
    save the predicted mask image to .jpg file
    save the predicted mask prediction to submission file using rle_encode
    """
    if torch.cuda.is_available():
        net.cuda()
    # net = nn.DataParallel(net)
    # net.load_state_dict(torch.load('models/unet.pth'))

    pred_labels = pred(test_loader, net)
    names = glob.glob(CARANA_DIR+'/test/*.jpg')
    names = [name.split('/')[-1][:-4] for name in names]
    # save mask
    save_mask(mask_imgs=pred_labels, model_name='unet-v1-768*1152', names=names)


def do_submisssion():
    mask_names = glob.glob(CARANA_DIR+'/unet-v1-768*1152/*.png')
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
    df.to_csv(CARANA_DIR+'/unet-v1-640*960.csv.gz', index=False, compression='gzip')


if __name__ == '__main__':
    net = UNetV1()
    # from scipy.misc import imshow
    valid_loader, train_loader = get_valid_dataloader(6, H=768, W=1152), \
                               get_train_dataloader(H=768, W=1152, batch_size=2, preload=True, num_works=4)
    train(net)
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
    # test_loader = get_test_dataloader(batch_size=8, H=640, W=960)
    # save_mask(mask_imgs=pred_labels, model_name='unet', names=names)

    # test(net)
    # do_submisssion()
