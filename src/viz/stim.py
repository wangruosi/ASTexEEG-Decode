import os
import numpy as np
import matplotlib.pyplot as plt

from ..configs import data_dir


def load_stim(stim_type, condition):
    '''load stimulus images with cv2'''
    ims = []
    in_path = os.path.join(data_dir, 'imageset', stim_type, 'ImageFiles')
    files = sorted([f for f in os.listdir(in_path) if condition in f])
    for f in files:
        im = plt.imread(os.path.join(in_path, f))
        ims.append(im)
    return ims


def combine_im(ims, n_cols=3, n_rows=3):
    return np.vstack([np.hstack(ims[i_row*n_cols: (i_row+1)*n_cols])
                        for i_row in range(n_rows)])


def plot_stim_examples(stim_type, figsize=(4, 4), version='main', save=False):
    conds = ('BigAnimals', 'BigObjects',
             'SmallAnimals', 'SmallObjects')

    if version == 'main':
        n_cols, n_rows = 3, 3
        _, axes = plt.subplots(2, 2, figsize=figsize)
    elif version == 'suppl':
        n_cols, n_rows = 3, 5
        _, axes = plt.subplots(1, 4, figsize=figsize)

    axes = axes.flatten()

    for cond, ax in zip(conds, axes):
        ims = load_stim(stim_type, cond)
        combined = combine_im(ims, n_cols=n_cols, n_rows=n_rows)
        ax.imshow(combined, cmap=plt.get_cmap('gray'))
        ax.set_axis_off()

    plt.subplots_adjust(left=0, top=1, right=1,
                        bottom=0, wspace=.01, hspace=.01)
