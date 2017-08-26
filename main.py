import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, SGD
import glob
from dataset import get_valid_dataloader, get_train_dataloader, get_test_dataloader, CARANA_DIR
from unet import UNet512, UNetV2, UNetV3
from myunet import UNet_double_1024_5, UNet_1024_5, SoftDiceLoss
from util import pred, evaluate, dice_coeff, run_length_encode, save_mask
from myunet import BCELoss2d
import cv2
from scipy.misc import imread
import pandas as pd


EPOCH = 70
LEARNING_RATE = 5e-4
L2_DECAY = 7e-4


def lr_scheduler(optimizer, epoch):
    if 0 <= epoch <= 10:
        lr = 0.001
    elif 10 < epoch<= 35:
        lr = 0.001
    elif 35 < epoch <= 40:
        lr = 0.0009
    else:
        lr = 0.0005
    for param in optimizer.param_groups:
        param['lr'] = lr


def train(net):
    optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)  ###0.0005
    bce2d = BCELoss2d()
    softdice = SoftDiceLoss()
    if torch.cuda.is_available():
        net.cuda()

    # train
    best_val_loss = 0.0
    for e in range(EPOCH):
        # iterate over batches
        # lr_scheduler(optimizer, e)
        num = 0
        total = len(train_loader.dataset.img_names)
        for idx, (img, label) in enumerate(train_loader):
            net.train()
            num += img.size(0)
            img = Variable(img.cuda()) if torch.cuda.is_available() else Variable(img)
            # label = label.long()
            label = Variable(label.cuda()) if torch.cuda.is_available() else Variable(label)
            out, logits = net(img)
            # out = out.Long()
            loss = bce2d(out, label)
            # fresh gradients
            # optimizer.zero_grad()
            # do backward pass
            # loss.backward()
            # update
            # optimizer.step()

            if idx == 0:
                optimizer.zero_grad()
            loss.backward()
            if idx % 10 == 0:
                optimizer.step()
                optimizer.zero_grad()

            if idx % 10 == 0:
                print('\r {}: Training loss is'.format(num/total), loss.data[0], flush=True, end='')
        if e % 1 == 0:
            # validate
            pred_labels = pred(valid_loader, net)
            valid_loss = evaluate(valid_loader, net, bce2d)
            # print(pred_labels)
            dice = dice_coeff(preds=pred_labels, targets=valid_loader.dataset.labels)
            print('\nEpoch {}: validation loss-{}, dice coeff-{}, best loss-{}'.format(e, valid_loss, dice, best_val_loss))
            if best_val_loss < dice:
                print('Save')
                torch.save(net.state_dict(), 'models/unet_full_5000.pth')
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
    for (s, e) in [(20000, 40000), (40000, 60000), (60000, 80000), (80000, 100064)]:
        test_loader = get_test_dataloader(batch_size=8, H=640, W=960, start=s, end=e, out_h=1280, out_w=1920,
                                     mean=None, std=None)
        pred_labels = pred(test_loader, net)
        names = glob.glob(CARANA_DIR+'/test/*.jpg')[s:e]
        names = [name.split('/')[-1][:-4] for name in names]
        # save mask
        save_mask(mask_imgs=pred_labels, model_name='unet_full_5000', names=names)
        del pred_labels


def do_submisssion():
    mask_names = glob.glob(CARANA_DIR+'/unet_full_5000/*.png')
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
    df.to_csv(CARANA_DIR+'/unet_full_5000.csv.gz', index=False, compression='gzip')


if __name__ == '__main__':
    net = UNet_double_1024_5((3, 1280, 1920), 1)
    net = nn.DataParallel(net)
    # net.load_state_dict(torch.load('models/unet1024_5000.pth'))
    # from scipy.misc import imshow
    valid_loader, train_loader = get_valid_dataloader(split='valid-88', batch_size=6, H=640, W=960, out_h=1280, out_w=1920,
                                                       mean=None,
                                                     std=None), \
                                get_train_dataloader(split='train-5000', H=640, W=960, batch_size=4, num_works=6,out_h=1280, out_w=1920,
                                                     mean=None,
                                                     std=None)
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

    # save_mask(mask_imgs=pred_labels, model_name='unet', names=names)

    test(net)
    do_submisssion()
