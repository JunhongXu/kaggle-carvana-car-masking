import glob

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD

from dataset import get_valid_dataloader, get_train_dataloader, get_test_dataloader, CARANA_DIR, mean, std
from myunet import BCELoss2d
from myunet import UNet_double_1024_5
from util import pred, evaluate, dice_coeff, run_length_encode, save_mask

EPOCH = 60
LEARNING_RATE = 5e-4
L2_DECAY = 7e-4


def lr_scheduler(optimizer, epoch):
    if 0 <= epoch <= 10:
        lr = 0.001
    elif 10 < epoch<= 35:
        lr = 0.0009
    elif 35 < epoch <= 40:
        lr = 0.0005
    else:
        lr = 0.0001
    for param in optimizer.param_groups:
        param['lr'] = lr


def train():
    criterion = BCELoss2d()

    for i in range(5):
        net = UNet_double_1024_5((3, 512, 512), 1)
        net = nn.DataParallel(net)
        net.cuda()
        optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)  ###0.0005

        train_loader = get_train_dataloader('train-%s' % i, num_works=6, mean=mean[i], std=std[i], batch_size=10)
        valid_loader = get_valid_dataloader(batch_size=16, split='valid-%s' % i, mean=mean[i], std=std[i])
        print('Training on fold %s' % i)
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
                # label = label.long()
                label = Variable(label.cuda()) if torch.cuda.is_available() else Variable(label)
                out, logits = net(img)
                # out = out.Long()
                loss = criterion(out, label)
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
                    torch.save(net.state_dict(), 'models/unet1024_%s.pth' % format(i))
                    best_val_loss = dice


def test():
    """
    save the predicted mask image to .jpg file
    save the predicted mask prediction to submission file using rle_encode
    """
    for i in range(5):
        for (start, end) in [(0, 30000), (30000, 60000), (60000, 90000), (90000, 100064)]:
            test_loader = get_test_dataloader(start=start, end=end, batch_size=16, H=512, W=512, std=std[i], mean=mean[i])
            net = UNet_double_1024_5((3, 512, 512), 1)
            net = nn.DataParallel(net)
            net.cuda()
            net.load_state_dict(torch.load('models/unet1024_{}.pth'.format(i)))
            pred_label = pred(test_loader, net)
        # for (start, end) in [(0, 10000), (10001, 20000), (20001, 35000), (35001, 50000), (50001, 65000), (65001, 80000),
        #                      (80001, 95000), (95001, 100064)]:
        #     pred_labels_sum = np.empty((end-start, 1024, 1024), dtype=np.uint8)
        #     for i in range(5):
        #         # load nets
        #         print(i)
        #         test_loader = get_test_dataloader(start=start, end=end, batch_size=8, H=512, W=512, std=std[i], mean=mean[i])
        #         net = UNet_double_1024_5((3, 512, 512), 1)
        #         net = nn.DataParallel(net)
        #         net.cuda()
        #         net.load_state_dict(torch.load('models/unet1024_{}.pth'.format(i)))
        #         pred_label = pred(test_loader, net)
        #         pred_labels_sum = pred_labels_sum + pred_label
        #         del pred_label
        #
        #     pred_labels_sum = (pred_labels_sum > 2).astype(np.uint8)
        #     # if torch.cuda.is_available():
        #     #     net.cuda()
        #     # # net = nn.DataParallel(net)
        #     # # net.load_state_dict(torch.load('models/unet.pth'))
        #     #
        #     # pred_labels = pred(test_loader, net)
            names = glob.glob(CARANA_DIR+'/test/*.jpg')[start:end]
            names = [name.split('/')[-1][:-4] for name in names]
            # # save mask
            save_mask(mask_imgs=pred_label, model_name='unet_1024_{}'.format(i), names=names)
            del pred_label


def do_submisssion():
    from scipy.misc import imread
    mask_names = []
    for i in range(0, 5):
        mask_name = glob.glob(CARANA_DIR+'/unet_1024_{}/*.png'.format(i))
        mask_names.append(mask_name)

    names = []
    rle = []
    for img_idx in range(len(mask_name)):
        test_name = mask_name[img_idx].split('/')[-1][:-4]
        names.append(test_name+'.jpg')
        mask = np.zeros((1280, 1918))
        for index in range(5):
            mask += cv2.resize(imread(mask_names[index][img_idx]), (1918, 1280))

        mask = (mask>2).astype(np.uint8)
        print('\rProgress %.5f' % (img_idx/len(mask_name)), end='', flush=True)
        rle.append(run_length_encode(mask))
    df = pd.DataFrame({'img': names, 'rle_mask': rle})
    df.to_csv(CARANA_DIR+'/unet_1024_5fold.csv.gz', index=False, compression='gzip')


if __name__ == '__main__':
    # train()
    # test()
    do_submisssion()
