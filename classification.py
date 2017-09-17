import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models.resnet import resnet34, resnet50, model_zoo, model_urls
from cls_labels import idx2label, label2idx
from torch.optim import SGD
from dataset import CARANA_DIR, get_cls_train_dataloader, get_cls_valid_dataloader\
    , get_train_dataloader, get_valid_dataloader, transform3, transform2, transform1
from refinenet import RefineNetV4_1024, BasicBlock
from main import SoftIoULoss, BCELoss2d, calculate_weight, lr_scheduler
import numpy as np
from util import Logger, dice_coeff, pred
import time
import random
import glob
import cv2


PRE_TRAIN_MODEL_NAME = 'resnet34_car'
START_EPOCH = 0
END_EPOCH = 40
in_h, in_w, out_h, out_w = 512, 512, 512, 512
MODEL_NAME = 'pre_trained_refinev4_512'
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
print_it = 20
DEBUG = True


def lr_adjuster(optimizer, e):
    if 0 <= e < 10:
        lr = 0.01
    elif 10 <= e < 20:
        lr = 0.005
    elif 20 <= e < 30:
        lr = 0.0005
    else:
        lr = 0.0001

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
        gt = Variable(gt.long().cuda(), volatile=True)
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
    model_dict = model.state_dict()
    pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
    model_dict.update({key: pretrained_dict[key] for key in pretrained_dict.keys() if 'fc' not in key})
    model.load_state_dict(model_dict)
    return model


def pretrain():
    start = 15
    end = 40
    train_loader, valid_loader = get_cls_train_dataloader(), get_cls_valid_dataloader()
    net = pretrained_resnet34()
    net = nn.DataParallel(net).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    logger = Logger(str(PRE_TRAIN_MODEL_NAME))
    logger.write('-----------------------Network config-------------------\n')
    logger.write(str(net)+'\n')
    total_num = len(train_loader.dataset)

    best_val_loss = np.inf

    logger.write('EPOCH || CLS loss || Avg CLS loss || Train Acc || Val loss || Val Acc || Time\n')
    if start > 0:
        net.load_state_dict(torch.load('models/'+PRE_TRAIN_MODEL_NAME+'.pth'))

    if DEBUG:
        labels, eval_loss, eval_acc = evaluate(net, valid_loader, criterion)
        img_names = valid_loader.dataset.img_names
        for label, img_name in zip(labels, img_names):
            cv2.imshow('pic', cv2.imread(img_name))
            print(idx2label[label])
            cv2.waitKey()

    for e in range(start, end):
        tic = time.time()
        net.train()
        lr_adjuster(optimizer, e)
        num = 0
        moving_loss = 0.0
        for idx, (imgs, labels) in enumerate(train_loader):
            batch_size = labels.size(0)
            num += batch_size
            imgs = Variable(imgs.cuda())
            labels = Variable(labels.long().cuda())
            optimizer.zero_grad()
            preds = net(imgs)
            loss = criterion(preds, labels)
            loss.backward()
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
                  % (e, loss.data[0], moving_loss/(idx+1), train_acc, eval_loss, eval_acc, (tac - tic) / 60),
                  flush=True, end='')
            print('\n')
            logger.write('%s || %.5f || %.5f || %.4f || %.5f || %.4f || %.2f\n'
                         % (e, loss.data[0], moving_loss, train_acc, eval_loss, eval_acc, (tac - tic) / 60), False)
            logger.log(train_acc, eval_acc, time=(tac - tic) / 60, train_loss=loss.data[0], val_loss=eval_loss)

            if eval_loss < best_val_loss:
                best_val_loss = eval_loss
                torch.save(net.state_dict(), 'models/' + PRE_TRAIN_MODEL_NAME + '.pth')
    logger.save()
    logger.file.close()


def load_pretrained_refinenet():
    pretrain_dict = torch.load('models/{}.pth'.format(PRE_TRAIN_MODEL_NAME))
    net = nn.DataParallel(RefineNetV4_1024(BasicBlock, [3, 4, 6, 3]))
    state_dict = net.state_dict()
    state_dict.update({key: state_dict[key] for key in pretrain_dict if 'fc' not in key})
    return net


def train_seg():
    train_loader, valid_loader = get_train_dataloader(batch_size=BATCH_SIZE, split='train-4088', mean=None, std=None, H=in_h, W=in_w,
                                                      out_h=out_h, out_w=out_w), \
                                 get_valid_dataloader(batch_size=EVAL_BATCH_SIZE, H=in_h, W=in_w, out_h=out_h, out_w=out_w, split='valid-300')

    net = load_pretrained_refinenet().cuda()

    optimizer = SGD([param for param in net.parameters() if param.requires_grad], lr=0.001, momentum=0.9,
                    weight_decay=0.0005)
    bce2d = BCELoss2d()
    # softdice = SoftDiceLoss()
    softiou = SoftIoULoss()
    if torch.cuda.is_available():
        net.cuda()
    logger = Logger(str(MODEL_NAME))
    logger.write('-----------------------Network config-------------------\n')
    logger.write(str(net) + '\n')
    train_loader.dataset.transforms = transform3
    if START_EPOCH > 0:
        net.load_state_dict(torch.load('models/' + MODEL_NAME + '.pth'))
        logger.write(('------------Resume training %s from %s---------------\n' % (MODEL_NAME, START_EPOCH)))

    # train
    logger.write('EPOCH || BCE loss || Avg BCE loss || Train Acc || Val loss || Val Acc || Time\n')
    best_val_loss = 0.0
    for e in range(START_EPOCH, END_EPOCH):
        # iterate over batches
        lr_scheduler(optimizer, e)
        moving_bce_loss = 0.0

        if e > 10:
            # reduce augmentation
            train_loader.dataset.transforms = transform2

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
            weight = calculate_weight(label)
            bce_loss = bce2d(out, label, weight=weight)
            loss = bce_loss + softiou(out, label)
            moving_bce_loss += bce_loss.data[0]

            logits = logits.data.cpu().numpy() > 0.5
            train_acc = dice_coeff(np.squeeze(logits), label.data.cpu().numpy())

            if idx == 0:
                optimizer.zero_grad()
            loss.backward()
            if idx % 2 == 0:
                optimizer.step()
                optimizer.zero_grad()

            if idx % print_it == 0:
                smooth_loss = moving_bce_loss / (idx + 1)
                tac = time.time()
                print('\r %.3f || %.5f || %.5f || %.4f || ... || ... || % .2f'
                      % (num / total, bce_loss.data[0], smooth_loss, train_acc, (tac - tic) / 60),
                      flush=True, end='')

        if e % 1 == 0:
            # validate
            smooth_loss = moving_bce_loss / (idx + 1)
            pred_labels = pred(valid_loader, net)
            valid_loss = evaluate(valid_loader, net, bce2d)
            # print(pred_labels)
            dice = dice_coeff(preds=pred_labels, targets=valid_loader.dataset.labels)
            tac = time.time()
            print('\r %s || %.5f || %.5f || %.4f || %.5f || %.4f || %.2f'
                  % (e, bce_loss.data[0], smooth_loss, train_acc, valid_loss, dice, (tac - tic) / 60),
                  flush=True, end='')
            print('\n')
            logger.write('%s || %.5f || %.5f || %.4f || %.5f || %.4f || %.2f\n'
                         % (e, bce_loss.data[0], smooth_loss, train_acc, valid_loss, dice, (tac - tic) / 60), False)
            logger.log(train_acc, dice, time=(tac - tic) / 60, train_loss=bce_loss.data[0], val_loss=valid_loss)
            if best_val_loss < dice:
                torch.save(net.state_dict(), 'models/' + MODEL_NAME + '.pth')
                best_val_loss = dice
    logger.save()
    logger.file.close()


if __name__ == '__main__':
    # train_valid_split()
    # pretrain()
    train_seg()
