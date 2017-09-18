import torch
from dataset import CARANA_DIR, get_pesudo_train_dataloader, get_test_dataloader, get_valid_dataloader, get_train_dataloader,\
    transform1, transform2, transform3
import cv2
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import SGD
from main import BCELoss2d, SoftIoULoss, calculate_weight, lr_scheduler
import time
from util import Logger, dice_coeff, pred, evaluate
from torch.nn import functional as F
from scipy.misc import imsave
import glob
from refinenet import RefineNetV2_1024, RefineNetV4_1024, RefineNetV3_1024, BasicBlock, Bottleneck
from matplotlib import pyplot as plt
from PIL import Image

out_h, out_w = 1280, 1918
interval = 100
load_number = 3000
START_EPOCH = 12
END_EPOCH = 50
print_it = 20
EVAL_BATCH = 10


def build_ensembles():
    """Return four single best models to do ensemble"""
    net1 = nn.DataParallel(RefineNetV4_1024(BasicBlock, [3, 4, 6, 3])).cuda()
    net2 = nn.DataParallel(RefineNetV4_1024(BasicBlock, [3, 4, 6, 3])).cuda()
    net3 = nn.DataParallel(RefineNetV4_1024(BasicBlock, [3, 4, 6, 3])).cuda()
    net4 = nn.DataParallel(RefineNetV2_1024(Bottleneck, [3, 4, 6, 3])).cuda()
    net5 = nn.DataParallel(RefineNetV3_1024(Bottleneck, [3, 4, 6, 3])).cuda()

    # load models
    net1.load_state_dict(torch.load('models/refinenetv4_resnet34_1280*1920_hq.pth'))
    net2.load_state_dict(torch.load('models/refinenetv4_resnet34_1280*1280_hq.pth'))
    net3.load_state_dict(torch.load('models/refinenetv4_resnet34_1024*1024_hq.pth'))
    net4.load_state_dict(torch.load('models/refinenetv4_1024_hq.pth'))
    net5.load_state_dict(torch.load('models/refinenetv3_resnet50_1024*1024_hq.pth'))
    return (net1, net2, net3, net4, net5), ((1280, 1920), (1280, 1280), (1024, 1024), (1024, 1024), (1024, 1024))


def do_pesudo_label(ensembles, dims, debug=False):
    """Save averaged probabilities into folder train/pesudo_masks/"""
    # ensembles is a tuple of single best models
    times = load_number//interval
    # score_maps = torch.FloatTensor(len(scales), size, 1280, 1918)

    # predict and store the map for averaging
    for t in range(times):
        start = t * interval
        end = (t+1) * interval
        if t == (times - 1):
            end = load_number
        prob_maps = np.zeros((interval, out_h, out_w))
        names = []
        for idx, scale in enumerate(dims):
            print('\rScale: %s' % idx, flush=True, end='')
            H, W = scale
            upsample = None
            if H != out_h or W != out_w:
                upsample = nn.Upsample(size=(out_h, out_w), mode='bilinear')
            dataloader = get_test_dataloader(batch_size=10, H=H, W=W, start=start, end=end, out_h=out_h,
                                             load_number=load_number, out_w=out_w, mean=None, std=None)
            prev = 0
            # predict
            net = ensembles[idx]
            net.eval()
            single_model_pred = np.zeros((interval, out_h, out_w))
            for (img, name) in dataloader:
                batch_size = img.size(0)
                img = Variable(img, volatile=True).cuda()
                _, preds = net(img)
                if upsample is not None:
                    preds = upsample(preds)
                single_model_pred[prev:(prev+batch_size)] = np.squeeze(preds.data.cpu().numpy())
                prev += batch_size
                names += name
            prob_maps = single_model_pred + prob_maps
            del single_model_pred
            del dataloader
        prob_maps = prob_maps / len(ensembles)
        if debug:
            for prob, img in zip(prob_maps, names):
                print(img)
                img = cv2.imread(img)
                mask = (prob > 0.5).astype(np.uint8)
                mask = np.ma.masked_where(mask==0, mask)
                plt.subplot(1, 2, 1)
                plt.imshow(img)
                plt.imshow(mask, 'jet', alpha=0.6)
                plt.subplot(1, 2, 2)
                plt.imshow(prob)
                plt.show()
        save_pesudo_label(prob_maps, names)


def save_pesudo_label(probs, names):
    for prob, name in zip(probs, names):
        name = name.split('/')[-1][:-4]
        image = Image.fromarray(prob)
        image.save(CARANA_DIR + '/train/train_pesudo_masks/{}.tiff'.format(name))


def gen_split_indices():
    """
        Generate split indices from train/train_hq and test/pesudo_train and save in the split/train_pesudo file.
        Each row has a format of either train_hq/image_name or pesudo/image_name to be parsed in the dataset.
    """
    filename = CARANA_DIR + '/split/' + 'train_pesudo'

    # read the original split file
    with open(CARANA_DIR + '/split/' + 'train-4788') as f:
        content = f.readlines()
    orig_file = ['/train/train_hq/'+line.strip('\n') for line in content]

    # read the image names pesudo labeled by the ensemble
    pesudo_label_file =['/test_hq/'+name.split('/')[-1][:-5] for name in glob.glob(CARANA_DIR + '/train/' + 'train_pesudo_masks/*.tiff')]

    file = orig_file + pesudo_label_file
    # save these indices
    with open(filename, 'w') as f:
        for line in file:
            f.write(line+'\n')


def train():
    in_h, in_w, train_out_h, train_out_w = 1280, 1920, 1280, 1920
    # model_name = 'refinenetv4_resnet34_512*512'
    model_name = 'refinenetv4_resnet34_1280*1920_pesudo'
    train_loader = get_pesudo_train_dataloader(in_h, in_w, train_out_h, train_out_w, 2)
    # train_loader = get_train_dataloader(split='train-4788', H=in_h, W=in_w, batch_size=16, num_works=6,
    #                                                           out_h=train_out_h, out_w=train_out_w, mean=None, std=None)
    valid_loader = get_valid_dataloader(split='valid-300', batch_size=EVAL_BATCH, H=in_h, W=in_w,
                                                              preload=False, num_works=2, mean=None, std=None)

    net = RefineNetV4_1024(BasicBlock, [3, 4, 6, 3])
    # net.load_params('resnet34')
    net = nn.DataParallel(net)
    net.load_state_dict(torch.load('models/refinenetv4_resnet34_1280*1920_pesudo.pth'))
    optimizer = SGD([param for param in net.parameters() if param.requires_grad], lr=0.001, momentum=0.9,
                    weight_decay=0.0005)  ###0.0005
    bce2d = BCELoss2d()
    # softdice = SoftDiceLoss()
    softiou = SoftIoULoss()
    if torch.cuda.is_available():
        net.cuda()
    logger = Logger(str(model_name))
    logger.write('-----------------------Network config-------------------\n')
    logger.write(str(net) + '\n')
    train_loader.dataset.transforms = transform3
    if START_EPOCH > 0:
        net.load_state_dict(torch.load('models/' + model_name + '.pth'))
        logger.write(('------------Resume training %s from %s---------------\n' % (model_name, START_EPOCH)))
    # train
    logger.write('EPOCH || BCE loss || Avg BCE loss || Train Acc || Val loss || Val Acc || Time\n')
    best_val_loss = 0.0
    for e in range(START_EPOCH, END_EPOCH):
        # iterate over batches
        lr_scheduler(optimizer, e)
        moving_bce_loss = 0.0

        if e > 20:
            # reduce augmentation
            train_loader.dataset.transforms = transform2

        num = 0
        total = train_loader.dataset.num_sample
        tic = time.time()
        for idx, (img, label, name) in enumerate(train_loader):
            net.train()
            num += img.size(0)
            img = Variable(img.cuda()) if torch.cuda.is_available() else Variable(img)
            # label = label.long()
            label = Variable(label.cuda()) if torch.cuda.is_available() else Variable(label)
            out, logits = net(img)
            # out = out.Long()
            # if 'test' in name:
            weight = calculate_weight(label)
            # else:
            #     weight = None
            bce_loss = bce2d(out, label, weight=weight)
            loss = bce_loss + softiou(out, label)
            # loss = bce_loss
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
                torch.save(net.state_dict(), 'models/' + model_name + '.pth')
                best_val_loss = dice
    logger.save()
    logger.file.close()


def test():
    pass


def do_submission():
    pass


if __name__ == '__main__':
    # ensembles, dim = build_ensembles()
    # do_pesudo_label(ensembles, dim, debug=False)
    # split_test()
    # gen_split_indices()
    train()
