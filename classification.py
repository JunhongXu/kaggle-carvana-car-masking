import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models.resnet import resnet34, resnet50, model_zoo, model_urls
from cls_labels import idx2label, label2idx
from torch.optim import SGD
from dataset import CARANA_DIR, get_cls_train_dataloader, get_cls_valid_dataloader
import numpy as np
from util import Logger
import time
import random
import glob


MODEL_NAME = 'resnet34_car'
START_EPOCH = 0
END_EPOCH = 40


def lr_adjuster(optimizer, e):
    if 0 <= e < 10:
        lr = 0.01
    elif 10 <= e < 20:
        lr = 0.005
    elif 20 <= e < 30:
        lr = 0.001
    else:
        lr = 0.0005

    for param in optimizer.param_groups:
        param['lr'] = lr


def train_valid_split(num_valid=10000):
    images = glob.glob((CARANA_DIR + '/train/train_hq/*.jpg')) + glob.glob((CARANA_DIR + '/test_hq/*.jpg'))
    random.shuffle(images)
    train_img_names = images[:len(images) - num_valid]
    valid_img_names = images[len(images) - num_valid:]
    with open(CARANA_DIR+'/split/train-class', 'w') as f:
        for name in train_img_names:
            f.write(name+'\n')

    with open(CARANA_DIR+'/split/valid-class', 'w') as f:
        for name in valid_img_names:
            f.write(name+'\n')


def evaluate(net, dataloader, criterion):
    net.eval()
    total_size = len(dataloader.dataset)
    pred_labels = np.empty(total_size)
    prev = 0
    avg_loss = 0.0
    avg_acc = 0.0
    for idx, (img, gt) in enumerate(dataloader):
        batch_size = img.size(0)
        img = Variable(img.cuda(), volatile=True)
        gt = Variable(gt.cuda(), volatile=True)
        scores = net(img)
        loss = criterion(scores, gt)
        avg_loss += loss.data[0]

        scores = scores.data.cpu().numpy()
        labels = np.argmax(scores, axis=1)
        pred_labels[prev: prev+batch_size] = labels
        prev = prev + batch_size

        avg_acc += (labels == gt.data.cpu().numpy()).sum()/batch_size

    return pred_labels, avg_loss/(idx+1), avg_acc/(idx+1)


def pretrained_resnet34(num_classes=len(label2idx)):
    model = resnet34(False, num_classes=num_classes)
    model_dict = model.load_state_dict()
    pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
    model_dict.update({key: pretrained_dict[key] for key in pretrained_dict.keys() if 'fc' not in key})
    model.load_state_dict(model_dict)
    return model


def train():
    train_loader, valid_loader = get_cls_train_dataloader(), get_cls_valid_dataloader()
    # net = resnet34(pretrained=True, num_classes=len(label2idx))
    net = pretrained_resnet34()
    net = nn.DataParallel(net).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    logger = Logger(str(MODEL_NAME))
    logger.write('-----------------------Network config-------------------\n')
    logger.write(str(net)+'\n')
    total_num = len(train_loader.dataset)
    tic = time.time()

    moving_loss = 0.0
    best_val_loss = np.inf

    logger.write('EPOCH || CLS loss || Avg CLS loss || Train Acc || Val loss || Val Acc || Time\n')
    for e in range(START_EPOCH, END_EPOCH):
        net.train()
        lr_adjuster(optimizer, e)
        num = 0
        for idx, (imgs, labels) in enumerate(train_loader):
            batch_size = labels.size(0)
            num += batch_size
            imgs = Variable(imgs.cuda())
            labels = Variable(labels.float().cuda())

            preds = net(imgs)
            loss = criterion(preds)
            optimizer.zero_grad()
            optimizer.step()

            moving_loss += loss.data[0]

            _preds = preds.data.cpu().numpy()
            _labels = labels.data.cpu().numpy()
            train_acc = (np.argmax(_preds, axis=1) == _labels).sum() / batch_size

            if idx%20 == 0:
                tac = time.time()
                print('\r %.3f || %.5f || %.5f || %.4f || ... || ... || % .2f'
                      % (num / total_num, loss.data[0], moving_loss/(idx+1), train_acc, (tac - tic) / 60),
                      flush=True, end='')

        if e % 1 == 0:
            labels, eval_loss, eval_acc = evaluate(net, valid_loader, criterion)

            tac = time.time()
            print('\r %s || %.5f || %.5f || %.4f || %.5f || %.4f || %.2f'
                  % (e, loss.data[0], moving_loss, train_acc, eval_loss, eval_acc, (tac - tic) / 60),
                  flush=True, end='')
            print('\n')
            logger.write('%s || %.5f || %.5f || %.4f || %.5f || %.4f || %.2f\n'
                         % (e, loss.data[0], moving_loss, train_acc, eval_loss, eval_acc, (tac - tic) / 60), False)
            logger.log(train_acc, eval_acc, time=(tac - tic) / 60, train_loss=loss.data[0], val_loss=eval_loss)

            if eval_loss < best_val_loss:
                best_val_loss = eval_loss
                torch.save(net.state_dict(), 'models/' + MODEL_NAME + '.pth')
    logger.save()
    logger.file.close()


if __name__ == '__main__':
    # train_valid_split()
    train()