import torch
from dataset import CARANA_DIR, CarvanaDataSet, get_test_dataloader
import cv2
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
from scipy.misc import imsave
import glob
from refinenet import RefineNetV2_1024, RefineNetV4_1024, RefineNetV3_1024, BasicBlock, Bottleneck
from matplotlib import pyplot as plt
from PIL import Image

out_h, out_w = 1280, 1918
interval = 100
load_number = 3000


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
    pass


def test():
    pass


def do_submission():
    pass


if __name__ == '__main__':
    # ensembles, dim = build_ensembles()
    # do_pesudo_label(ensembles, dim, debug=False)
    # split_test()
    gen_split_indices()
