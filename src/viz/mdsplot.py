import os
import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import MDS

from ..configs import RANDOM_STATE, data_dir
from ..eeg.data import GroupModel


def plot_cross_decoding_viz(category, metric='euclid', timepoint=.18, 
                            random_state=RANDOM_STATE, s=12, figsize=(2, 2), save=False):

    original_slice, texform_slice = slice(0, 60), slice(60, 120)
    labels = dict(animacy=np.repeat([1, 0, 1, 0], 15),
                  size=np.repeat([1, 0], 30))
    colors = dict(animacy=('mediumorchid', 'forestgreen'),
                  size=('dodgerblue', 'darkorange'))

    # load rdm and average across group
    grp = GroupModel()
    results = grp.load('rdm_eeg', stim_type='all', metric='euclid', whiten=True)
    times = results['timepoint'].squeeze()
    tid = np.where(times==timepoint)[0][0]
    rdm = results['rdm'][tid]

    mds = MDS(n_components=2, dissimilarity='precomputed',
              random_state=random_state)
    transformed = mds.fit_transform(rdm)

    # extract mds data|
    X_tex = transformed[texform_slice]
    y_tex = labels[category]
    X_orig = transformed[original_slice]
    y_orig = labels[category]

    # train classifier for texform images
    _, ax = plt.subplots(figsize=figsize)
    c1, c2 = colors[category]

    # texform
    tex_kwargs = dict(alpha=0.6, s=s, linewidths=0)
    ax.scatter(X_tex[y_tex == 1, 0], X_tex[y_tex == 1, 1],
               color=c1, **tex_kwargs)
    ax.scatter(X_tex[y_tex == 0, 0], X_tex[y_tex == 0, 1],
               color=c2, **tex_kwargs)

    orig_kwargs = dict(s=s, facecolor='none', linewidths=.8, alpha=0.6)
    ax.scatter(X_orig[y_orig == 1, 0],
               X_orig[y_orig == 1, 1], color=c1, **orig_kwargs)
    ax.scatter(X_orig[y_orig == 0, 0],
               X_orig[y_orig == 0, 1], color=c2, **orig_kwargs)

    # means
    ax.scatter(*X_tex[y_tex == 1].mean(0), facecolor=c1, edgecolor='black',
               linewidths=.2, s=s*3)
    ax.scatter(*X_tex[y_tex == 0].mean(0), facecolor=c2, edgecolor='black',
               linewidths=.2, s=s*3)
    ax.scatter(*X_orig[y_orig == 1].mean(0),
               facecolor='white', edgecolor=c1, s=s*3)
    ax.scatter(*X_orig[y_orig == 0].mean(0),
               facecolor='white', edgecolor=c2, s=s*3)
#     del plot_kwargs

    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1)


def plot_tri_decoding_viz(stim_type, animacy, metric='euclid', timepoint=.18,
                          random_state=RANDOM_STATE, s=3, figsize=(3, 3), ax=None, save=False):

    n_exemplar = 15
    xmin, ymin, xmax, ymax = -20, -20, 20, 20

    mask_mappings = {('original', 'animate'): np.concatenate([np.arange(0, 15), np.arange(30, 45)]),
                     ('original', 'inanimate'): np.concatenate([np.arange(15, 30), np.arange(45, 60)]),
                     ('texform', 'animate'): np.concatenate([np.arange(60, 75), np.arange(90, 105)]),
                     ('texform', 'inanimate'): np.concatenate([np.arange(75, 90), np.arange(105, 120)])
                     }
    color_mappings = {'animate': ('mediumpurple', 'palevioletred'),
                      'inanimate': ('lightskyblue', 'orange')
                      }
    plotting_mappings = {'original': dict(lw=1, alpha=.5, fill=None),
                         'texform': dict(lw=0, alpha=.2)
                         }

    mask_ = mask_mappings.get((stim_type, animacy))
    colors = np.repeat(color_mappings.get(animacy), n_exemplar)

    # load rdm and average across group
    grp = GroupModel()
    results = grp.load('rdm_eeg', stim_type='all', metric='euclid', whiten=True)
    times = results['timepoint'].squeeze()
    tid = np.where(times == timepoint)[0][0]
    rdm = results['rdm'][tid]

    # 2D projection
    mds = MDS(n_components=2, dissimilarity='precomputed',
              random_state=random_state)
    xx, yy = np.meshgrid(mask_, mask_)
    transformed = mds.fit_transform(rdm[xx, yy])
    transformed *= 15

    # load icons
    icons = np.concatenate([_load_icons(), _load_icons()])

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    for ii in range(n_exemplar*2):
        im = icons[mask_][ii]
        extent = _get_extent(im.shape[:2], transformed[ii, :], shrink=60)

        circle = plt.Circle(transformed[ii, :], 2.3, color=colors[ii],
                            **plotting_mappings[stim_type])
        ax.add_patch(circle)
        ax.imshow(rgb2gray(im), extent=extent, cmap=plt.cm.binary)

    ax.set_axis_off()
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    plt.tight_layout()


def _get_extent(img_shape, pos, shrink=100):
    h, w = img_shape
    x, y = pos
    return (np.round(x - w/shrink),
            np.round(x + w/shrink),
            np.round(y - h/shrink),
            np.round(y + h/shrink))


def _load_icons():
    return np.load(os.path.join(data_dir, 'imageset', 'icons.npy'),
                   allow_pickle=True)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])