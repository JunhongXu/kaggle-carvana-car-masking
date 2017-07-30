import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, SGD
import glob
from dataset import get_valid_dataloader, get_train_dataloader, get_test_dataloader, CARANA_DIR
from unet import UNet
from util import pred, evaluate, dice_coeff, rle_encode, save_mask
import numpy as np


EPOCH = 50
LEARNING_RATE = 1e-4
L2_DECAY = 5e-4


def lr_scheduler(optimizer, epoch):
    if 0 < epoch <=10:
        lr = 0.1
    elif 10 < epoch <= 25:
        lr = 0.01
    elif 25 < epoch <=35:
        lr = 0.005
    elif 35 < epoch <= 40:
        lr = 0.0005
    else:
        lr = 0.0001
    for param in optimizer.param_groups:
        param['lr'] = lr


def train(net):
    # valid_loader = get_valid_dataloader(20)
    optimizer = Adam(params=net.parameters(), lr=LEARNING_RATE, weight_decay=L2_DECAY)
    # optimizer = SGD(params=net.parameters(), lr=LEARNING_RATE, weight_decay=L2_DECAY, momentum=0.95)
    criterion = nn.NLLLoss2d()
    if torch.cuda.is_available():
        net.cuda()
    net = nn.DataParallel(net)
    print(net)
    # train
    best_val_loss = np.inf
    for e in range(EPOCH):
        # iterate over batches
        net.train()
        for idx, (img, label) in enumerate(train_loader):
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
                print(loss.data[0])

        if e % 1 == 0:
            # validate
            logits = pred(valid_loader, net)
            valid_loss = evaluate(valid_loader, net, criterion)
            pred_labels = np.argmax(logits, axis=1)
            # print(pred_labels)
            dice = dice_coeff(preds=pred_labels, targets=valid_loader.dataset.labels)
            print(valid_loss, dice)
            if best_val_loss < dice:
                torch.save(net.state_dict(), 'models/unet.pth')
                best_val_loss = dice


def test(net):
    """
    save the predicted mask image to .jpg file
    save the predicted mask prediction to submission file using rle_encode
    """
    if torch.cuda.is_available():
        net.cuda()
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load('models/unet.pth'))

    logtis = pred(test_loader, net)
    pred_mask = np.argmax(logtis, axis=1).astype(np.uint8)
    names = glob.glob(CARANA_DIR+'/test/*.jpg')
    names = [name.split('/')[-1][:-4] for name in names]
    # save mask
    save_mask(mask_imgs=pred_mask, model_name='unet', names=names)


if __name__ == '__main__':
    net = UNet()
    # train_loader, valid_loader = get_train_dataloader(20), get_valid_dataloader(64)
    # train(net)

    test_loader = get_test_dataloader(128)
    test(net)
