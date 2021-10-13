
import os
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import squareform

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from ..configs import RANDOM_STATE, data_dir
from ..eeg.data import GroupModel
from ..stats.scoring import get_sig_timepoints


DECODE_MAPPINGS = {
    ('original', False): dict(label='original', color='silver', dash=''),
    ('texform', False): dict(label='texform', color='black', dash=''),
    ('texform', True): dict(label='cross', color='dimgray', dash=(1, 1)),
    ('original', True): dict(label='original=>texform', color='dimgray', dash=(4, 2)),
    ('original_animal', False): dict(label='big vs small animals', color='mediumorchid', dash=''),
    ('original_object', False): dict(label='big vs small objects', color='forestgreen', dash=''),
    ('texform_animal', False): dict(label='big vs small animals', color='mediumorchid', dash=''),
    ('texform_object', False): dict(label='big vs small objects', color='forestgreen', dash=''),
}

AXES_PARAMS = {
    'large': dict(
        ylim=(.38, .82),
        yticks=np.arange(.4, .82, .05),
        sig_start_y=.81,
        sig_unit=.01,
        stimulus_polygon=[[0, .384], [.1, .396], [.32, .396], [0.4, .384]],
        stimulus_position=(.2, .404)
    ),
    'small': dict(
        ylim=(.44, .66),
        yticks=np.arange(.4, .7, .05),
        sig_start_y=.655,
        sig_unit=.005,
        stimulus_polygon=[[0, .442], [.1, .448], [.32, .448], [0.4, .442]],
        stimulus_position=(.2, .452)
    ),
    'medium': dict(
        ylim=(.44, .73),
        yticks=np.arange(.4, .72, .05),
        sig_start_y=.72,
        sig_unit=.006,
        stimulus_polygon=[[0, .442], [.1, .448], [.32, .448], [0.4, .442]],
        stimulus_position=(.2, .452)
    )
}


def plot_category_decoding(category, args_list, **kwargs):
    if category == 'animacy' and ('original', False) in args_list:
        yaxis_range = 'large'
    else:
        yaxis_range = 'small'
    make_decoding_plot(_make_decoding_df('category', args_list, category=category),
                       palette=_get_param_dict(args_list, 'color'),
                       dashes=_get_param_dict(args_list, 'dash'),
                       yaxis_range=yaxis_range, **kwargs)


def plot_tri_category_decoding(stim_type, **kwargs):
    args_list = [(f'{stim_type}_animal', False),
                 (f'{stim_type}_object', False)]
    alpha = .5 if stim_type == 'original' else 1
    make_decoding_plot(_make_decoding_df('category', args_list, category='size'),
                       palette=_get_param_dict(args_list, 'color'),
                       dashes=_get_param_dict(args_list, 'dash'), compare=True,
                       yaxis_range='medium', alpha=alpha, **kwargs)


def plot_pair_decoding(cross=False, **kwargs):
    args_list = [('original', cross),
                 ('texform', cross)]
    make_decoding_plot(_make_decoding_df('pair', args_list),
                       palette=_get_param_dict(args_list, 'color'),
                       dashes=_get_param_dict(args_list, 'dash'),
                       compare=True, yaxis_range='medium', **kwargs)



def make_decoding_plot(df, palette=None, dashes=None, legend=False, alpha=1, 
                       against_chance = True, compare=False, ref_score=None,
                       yaxis_range='small', ax=None, figsize=(6, 4)):

    n_subs = len(df['subject'].unique())
    times = df['time'].unique()
    labels = df['label'].unique()
    params = AXES_PARAMS[yaxis_range]

    if not ax:
        _, ax = plt.subplots(figsize=figsize)

    # draw lineplot
    sns.lineplot(x='time', y='y', data=df, ax=ax, alpha=alpha,
                 hue='label', palette=palette,
                 style='label', dashes=dashes,
                 linewidth=1, err_kws=dict(linewidth=0, alpha=.2))

    if not legend:
        ax.get_legend().remove()

    if ref_score:
        ax.plot(times, ref_score, color='dimgray', alpha=alpha, lw=.7)

    ax.axhline(0.5,  color='black')
    ax.set(xlim=(-.1, .9), xlabel='time (s)',
           ylabel='classification\n(ROC AUC)',
           xticks=np.arange(0, .9, .1),
           yticks=params['yticks'],
           ylim=params['ylim'])

    plt.tight_layout()
    sns.despine(trim=True, ax=ax)

    if against_chance:
        for ii, label in enumerate(labels):
            scores = df.loc[df['label'] == label, 'y'].values.reshape(n_subs, -1)
            sig_points = get_sig_timepoints(
                scores - .5, times, alternative='greater')
            ax.scatter(sig_points,
                    np.ones(len(sig_points)) *
                    params['sig_start_y'] - ii * params['sig_unit'],
                    color=palette.get(label), alpha=alpha, s=1, marker='s')

    if compare:
        for ii, (l1, l2) in enumerate(combinations(labels, 2)):
            s1 = df.loc[df['label'] == l1, 'y'].values.reshape(n_subs, -1)
            s2 = df.loc[df['label'] == l2, 'y'].values.reshape(n_subs, -1)
            sig_points = get_sig_timepoints(s1 - s2, times)
            ax.scatter(sig_points,
                    np.ones(len(sig_points)) * .48 - ii * params['sig_unit'],
                    color='gray', alpha=alpha, s=1, marker='s')

    ax.add_patch(Polygon(params['stimulus_polygon'], closed=True,
                         facecolor='k', edgecolor='none', alpha=.8))
    ax.text(*params['stimulus_position'],
            'stimulus', horizontalalignment='center')




def _make_decoding_df(decode_type, args_list, **kwargs):
    grp = GroupModel()
    dfs = []
    for (stim_type, cross) in args_list:
        results =grp.load(f'decode_{decode_type}', stim_type=stim_type, cross=cross, **kwargs)
        scores, times = results['score'], results['timepoint'].squeeze()
        
        dfs.append(
            make_df(scores, times, label=DECODE_MAPPINGS[(
                stim_type, cross)]['label'])
        )
    return pd.concat(dfs, ignore_index=True)


def _get_param_dict(args_list, param):
    return {DECODE_MAPPINGS[args]['label']:
            DECODE_MAPPINGS[args][param] for args in args_list}


def make_df(scores, times, label=None):
    '''Make Dataframe'''
    n_sub, n_times = scores.shape
    subjects = [f'Sub{i+1:01d}' for i in range(n_sub)]

    df = pd.DataFrame(zip(scores.flatten(),
                          np.tile(times, n_sub),
                          np.repeat(subjects, n_times)),
                      columns=['y', 'time', 'subject'])

    if label:
        df['label'] = label
    return df









