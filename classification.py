import torch
import torch.nn as nn
from torch.autograd import Variable
from dataset import CARANA_DIR
import numpy as np
import random
import glob


def train_valid_split(num_valid=10000):
    images = glob.glob((CARANA_DIR + '/train/train_hq/*.jpg')) + glob.glob((CARANA_DIR + '/test_hq/*.jpg'))
    random.shuffle(images)
    train_img_names = images[:len(images) - num_valid]
    valid_img_names = images[len(images) - num_valid:]
    with open(CARANA_DIR+'/split/train-class') as f:
        for name in train_img_names:
            f.write(name+'\n')

    with open(CARANA_DIR+'/split/valid-class') as f:
        for name in valid_img_names:
            f.write(name+'\n')
