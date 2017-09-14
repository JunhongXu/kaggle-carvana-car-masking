import glob
import cv2
from dataset import CARANA_DIR
import numpy as np
from math import ceil
from util import save_mask
from matplotlib import pyplot as plt
from scipy.misc import imread
import pandas as pd
from util import run_length_encode

ensembles = [
                'refinenetv4_resnet34_1280*1920_hq',
                'refinenetv4_resnet34_1280*1280_hq',    # 0.9967
                'refinenetv4_resnet34_1024*1024_hq',    # 0.9966
                'refinenetv4_1024_hq',                  # 0.9965
                'refinenetv3_resnet50_1024*1024_hq'     # 0.9965
            ]
NAME = 'ensemble-2'
H, W = 1280, 1918
interval = 1000
TOTAL_TEST = 100064
DEBUG = False


# get results
def read_imgs(image_names):
    N = len(image_names)
    images = np.empty((N, H, W), np.uint8)
    for idx, img_name in enumerate(image_names):
        image = imread(img_name)
        image = cv2.resize(image, (W, H)).astype(np.uint8)
        images[idx] = image
    return images


# ensemble
def do_ensemble():
    ensemble_len = len(ensembles)
    ensemble_img_names = []
    times = ceil(TOTAL_TEST/interval)
    for e in ensembles:
        names = sorted(glob.glob(CARANA_DIR+'/{}/*.png'.format(e)))
        ensemble_img_names.append(names)
    for t in range(0, times):
        print('\r[!]Process: %.4f' % (t/times), end='', flush=True)
        start = t * interval
        end = (t+1) * interval
        if t == (times - 1):
            end = TOTAL_TEST

        current_names = [e[start:end] for e in ensemble_img_names]
        labels = np.zeros((len(current_names[0]), H, W), dtype=np.uint8)
        for ensemble_idx in range(ensemble_len):
            current_images = current_names[ensemble_idx]
            images = read_imgs(current_images)
            if ensemble_idx == 0:
                images = images * 1
            labels = labels + images
            del images
        labels = (labels > (ensemble_len // 2)).astype(np.uint8)

        names = [name.split('/')[-1][:-4] for name in current_names[0]]
        if DEBUG:
            images = ['{}/test_hq/{}.jpg'.format(CARANA_DIR, n) for n in names]
            for l, img in zip(labels, images):
                mask = cv2.resize((l).astype(np.uint8),  (1918, 1280))
                mask = np.ma.masked_where(mask==0, mask)
                plt.imshow(cv2.resize(cv2.imread(img), (1918, 1280)))
                plt.imshow(mask, 'jet', alpha=0.6)
                print(img)
                plt.show()


        # save labels
        save_mask(mask_imgs=labels, model_name=NAME, names=names)
        del labels

def do_submisssion():
    mask_names = glob.glob(CARANA_DIR+'/'+NAME+'/*.png')
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
    df.to_csv(CARANA_DIR+'/'+NAME+'.csv.gz', index=False, compression='gzip')


# divide results
if __name__ == '__main__':
    do_ensemble()
    do_submisssion()
