import sys
import cv2
import numpy as np
from scipy.misc import imread
import glob
from dataset import CARANA_DIR
from ensembles_majority_vote import ensembles, read_imgs
from matplotlib import pyplot as plt


# utility function
def plot_mask(img_names, mask_names, titles):
    plt.figure(figsize=(20, 15))
    mask_1, mask_2 = mask_names
    t1, t2 = titles
    num_col = 2
    num_row = 2
    N = len(img_names)
    idxs = np.random.randint(0, N, num_col*num_row//2)
    idxs = np.tile(idxs, 2)
    for i, idx in enumerate(idxs):
        image = cv2.imread(img_names[idx])
        retrieved_mask = mask_1 if i < 2 else mask_2
        title = t1 if i < 2 else t2
        mask = cv2.resize(imread(retrieved_mask[idx]),  (1918, 1280))
        mask = np.ma.masked_where(mask==0, mask)
        plt.subplot(num_row, num_col, i+1)
        plt.title(title)
        plt.imshow(cv2.resize(image, (1918, 1280)))
        plt.imshow(mask, 'jet', alpha=0.6)
        plt.gca().axis('off')
    plt.show()

if __name__ == '__main__':
    ensemble_v1_dir = CARANA_DIR + '/ensemble-1'
    single_models_dirs = [CARANA_DIR + '/{}/'.format(model) for model in ensembles]
    print('Ensemble directory: %s' % ensemble_v1_dir)
    print('Single models:')
    print('\n'.join(single_models_dirs))


    # get all models' image names
    test_img_names = sorted(glob.glob(CARANA_DIR+'/test_hq/*.jpg'))
    ensemble_mask_names = sorted(glob.glob(ensemble_v1_dir+'/*.png'))
    model_mask_names = [sorted(glob.glob(mask_dir+'*.png')) for mask_dir in single_models_dirs]
    print(len(ensemble_mask_names))
    print(len(model_mask_names[0]))
    print(len(test_img_names))

    # ensemble v.s. best model
    # plot_mask(test_img_names,( ensemble_mask_names,  model_mask_names[0]), ['ensemble', 'best single model'])

    # best model v.s. second best
    # plot_mask(test_img_names, (model_mask_names[1], model_mask_names[2]), ['best', 'second best'])

    # plot_mask(test_img_names, (model_mask_names[2], model_mask_names[3]), ['second best', 'third best'])
