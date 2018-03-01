import glob
import time
from math import ceil

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from scipy.misc import imread
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import SGD

from dataset import transform2, transform3, get_valid_dataloader, get_train_dataloader, get_test_dataloader, CARANA_DIR
from myunet import BCELoss2d, SoftIoULoss, FocalLoss
from refinenet import RefineNetV5_1024, RefineNetV3_1024, RefineNetV4_1024, RefineNetV2_1024, Bottleneck, BasicBlock
from util import Logger
from util import pred, evaluate, dice_coeff, run_length_encode, save_mask, calculate_weight
import tensorboardX


torch.manual_seed(0)
torch.cuda.manual_seed(0)
EPOCH = 40
START_EPOCH = 0
in_h = 1280
in_w = 1920
out_w = 1920
out_h = 1280
print_it = 30
interval = 500
NUM = 100064
USE_WEIGHTING = True
model_name = 'refinenetv4_resnet34_1280*1920_focal_loss_hq'
BATCH = 32
EVAL_BATCH = 32
DEBUG = False
is_training = True
MULTI_SCALE = False
scales = [(1280, 1920), (1280, 1536), (1280, 1280), (1024, 1024), (1024, 1024), (1024, 1024)]
model_names = ['refinenetv4_1280_1920', 'refinenetv4_1280_1536', 'refinenetv4_1280_1280',
               'refinenetv2_1024_1024', 'refinenetv5_1024_1024', 'refinenetv3_1024_1024']


def lr_scheduler(optimizer, epoch):
    if 0 <= epoch <= 10:
        lr = 0.01
    elif 10 < epoch <= 25:
        lr = 0.005
    elif 25 < epoch <= 50:
        lr = 0.001
    else:
        lr = 0.0005
    for param in optimizer.param_groups:
        param['lr'] = lr


def train():
    for name, scale in zip(model_names, scales):
        if 'v4' in name:
            model = RefineNetV4_1024(BasicBlock, [3, 4, 6, 3])
            model.load_params('resnet34')

        elif 'v2' in name:
            model = RefineNetV2_1024(Bottleneck, [3, 4, 6, 3])
            model.load_params('resnet50')

        elif 'v3' in name:
            model = RefineNetV3_1024(BasicBlock, [3, 4, 6, 3])
            model.load_params('resnet50')

        else:
            model = RefineNetV5_1024()
            model.load_vgg16()

        valid_loader, train_loader = get_valid_dataloader(split='valid-300', batch_size=EVAL_BATCH,
                                                          H=scale[0], W=scale[1],
                                                          out_h=scale[0], out_w=scale[1],
                                                          preload=False, num_works=2,
                                                          mean=None, std=None), \
                                         get_train_dataloader(split='train-4788', H=scale[0], W=scale[1],
                                                              batch_size=BATCH, num_works=6,
                                                              out_h=scale[0], out_w=scale[1],
                                                              mean=None, std=None,
                                                              transforms=transform3,
                                                              num=None)
        model = nn.DataParallel(model)
        model.cuda()
        optimizer = SGD([param for param in net.parameters() if param.requires_grad],
                        lr=0.001, momentum=0.9, weight_decay=0.0005)
        bce2d = BCELoss2d()
        focalloss = FocalLoss()
        softiou = SoftIoULoss()

        logger = Logger(str(name))
        logger.write('-----------------------Network config-------------------\n')
        logger.write(str(net)+'\n')
        train_loader.dataset.transforms = transform3
        batch_per_epoch = len(train_loader)
        print('Batch per epoch is %i' % batch_per_epoch)
        if START_EPOCH > 0:
            net.load_state_dict(torch.load('models/'+name+'.pth'))
            logger.write(('------------Resume training %s from %s---------------\n' %(name, START_EPOCH)))
        # train
        logger.write('EPOCH || BCE loss || Avg BCE loss || Train Acc || Val loss || Val Acc || Time\n')
        best_val_loss = 0.0
        writer = tensorboardX.SummaryWriter('tensorboard/%s' % name)
        for e in range(START_EPOCH, EPOCH):
            # iterate over batches
            lr_scheduler(optimizer, e)
            moving_bce_loss = 0.0

            if e > 10:
                # reduce augmentation
                train_loader.dataset.transforms = transform2

            num = 0
            total = train_loader.dataset.num if train_loader.dataset.num is not None else len(train_loader.dataset.img_names)
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
                focal = focalloss(logits, label)
                loss = bce_loss + softiou(out, label) + focal
                moving_bce_loss += bce_loss.data[0]

                logits = logits.data.cpu().numpy() > 0.5
                train_acc = dice_coeff(np.squeeze(logits), label.data.cpu().numpy())
                writer.add_scalar('%s/train/loss' % name, loss.data[0], global_step=e * batch_per_epoch + idx)
                writer.add_scalar('%s/train/dice' % name, train_acc, global_step=e * batch_per_epoch + idx)
                optimizer.zero_grad()
                # do backward pass
                loss.backward()
                # update
                optimizer.step()
                #
                # if idx == 0:
                #     optimizer.zero_grad()
                # loss.backward()
                # if idx % 5 == 0:
                #     optimizer.step()
                #     optimizer.zero_grad()

                if idx % print_it == 0:
                    smooth_loss = moving_bce_loss/(idx+1)
                    tac = time.time()
                    print('\r %.3f || %.5f || %.5f || %.4f || ... || ... || % .2f'
                          % (num/total, bce_loss.data[0], smooth_loss, train_acc, (tac-tic)/60),
                          flush=True, end='')

            if e % 1 == 0:
                # validate
                smooth_loss = moving_bce_loss/(idx+1)
                pred_labels = pred(valid_loader, net, upsample=False)
                valid_loss = evaluate(valid_loader, net, bce2d)
                # print(pred_labels)
                dice = dice_coeff(preds=pred_labels, targets=valid_loader.dataset.labels)
                tac = time.time()
                print('\r %s || %.5f || %.5f || %.4f || %.5f || %.4f || %.2f'
                      % (e, bce_loss.data[0], smooth_loss, train_acc, valid_loss, dice, (tac-tic)/60),
                      flush=True, end='')
                print('\n')
                logger.write('%s || %.5f || %.5f || %.4f || %.5f || %.4f || %.2f\n'
                             % (e, bce_loss.data[0], smooth_loss, train_acc, valid_loss, dice, (tac-tic)/60), False)
                logger.log(train_acc, dice, time=(tac-tic)/60, train_loss=bce_loss.data[0], val_loss=valid_loss)
                if best_val_loss < dice:
                    torch.save(net.state_dict(), 'models/'+name+'.pth')
                    best_val_loss = dice
                writer.add_scalar('%s/valid/loss' % name, valid_loss.data[0], e)
                writer.add_scalar('%s/valid/dice' % name, dice, e)
        logger.save()
        logger.file.close()


def test(net):
    """
    save the predicted mask image to .jpg file
    save the predicted mask prediction to submission file using rle_encode
    """
    # test
    upsampler = nn.Upsample(size=(out_h, out_w), mode='bilinear')
    net.eval()
    net.load_state_dict(torch.load('models/'+model_name+'.pth'))
    if torch.cuda.is_available():
        net.cuda()

    times = ceil(NUM/interval)

    for t in range(0, int(times)):
        print('Time ', t/times)
        s = t * interval
        e = (t+1) * interval
        if t == (times -1):
            e = NUM
        test_loader = get_test_dataloader(batch_size=EVAL_BATCH, H=in_h, W=in_w, start=s, end=e, out_h=out_h,
                                          out_w=out_w, mean=None, std=None)
        # if MULTI_SCALE:
        #     pred_labels = multi_scale(net, test_loader, s, e)
        # else:
        pred_labels = pred(test_loader, net, verbose=not DEBUG, upsample=upsampler)

        if DEBUG:
            for l, img in zip(pred_labels, test_loader.dataset.img_names):
                mask = cv2.resize((l).astype(np.uint8),  (1918, 1280))
                mask = np.ma.masked_where(mask==0, mask)
                plt.imshow(cv2.resize(cv2.imread(img), (1918, 1280)))
                plt.imshow(mask, 'jet', alpha=0.6)
                print(img)
                plt.show()
        names = test_loader.dataset.img_names
        names = [name.split('/')[-1][:-4] for name in names]
        # save mask
        save_mask(mask_imgs=pred_labels, model_name=model_name+'_upsample_preds', names=names)
        del pred_labels


def do_submisssion():
    mask_names = glob.glob(CARANA_DIR+'/'+model_name+'_upsample_preds'+'/*.png')
    names = []
    rle = []
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
    net = RefineNetV4_1024(BasicBlock, [3, 4, 6, 3])
    net.load_params('resnet34')
    net = nn.DataParallel(net).cuda()
    if 0:
        net.load_state_dict(torch.load('models/{}.pth'.format(model_name)))
        img = cv2.resize(cv2.imread('/media/jxu7/BACK-UP/Data/carvana/test_hq/000aa097d423_04.jpg'), (1024, 1024))
        img_ = Variable(torch.from_numpy(img.reshape(1, 1024, 1024, 3).transpose(0, 3, 1, 2)/255.), volatile=True).cuda()
        img_ = img_.float()
        l, mask = net(img_)
        mask = (mask.data.cpu().numpy() > 0.5).astype(np.uint8)
        print(mask.sum())
        cv2.imshow('frame', mask.reshape(1024, 1024)*100)
        cv2.imshow('orig', img.reshape(1024, 1024, 3))
        cv2.waitKey()
    if is_training:
        valid_loader, train_loader = get_valid_dataloader(split='valid-300', batch_size=EVAL_BATCH, H=in_h, W=in_w,
                                                          out_h=out_h, out_w=out_w,
                                                          preload=False, num_works=2, mean=None, std=None), \
                                         get_train_dataloader(split='train-4788', H=in_h, W=in_w, batch_size=BATCH, num_works=6,
                                                              out_h=out_h, out_w=out_w, mean=None, std=None, transforms=transform3, num=None)
        train()
    else:
        test(net)
        do_submisssion()
