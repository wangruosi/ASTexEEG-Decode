
import os
from collections import defaultdict

import numpy as np
from detecta import detect_peaks
from scipy.stats import wilcoxon
from scipy.io import savemat

from ..configs import result_dir, RANDOM_STATE, T_RANGE
from ..eeg.data import GroupModel


N_ITERS = 5000

def get_latencies(category): 

    decodings = {
        'original': ('original', False),    # original => original  
        'texform': ('texform', False),      # texform  => texform   
        'cross': ('texform', True),         # texform  => original  
        'cross_suppl': ('original', True)  # original => texform
    }

    results, grp = defaultdict(), GroupModel()

    for label, (stim_type, cross)in decodings.items():
        # load decoding results
        data = grp.load('decode_category', stim_type=stim_type, category=category, cross=cross)
        scores, times = data['score'], data['timepoint'].squeeze()

        results[f'{label}_score'] = scores

        # group latencies
        results[f'{label}_onset'] = find_onset(scores, times)
        results[f'{label}_peak'] = find_peak(scores, times)

        # bootstrapped latencies
        results[f'{label}_boot_onset']= get_bootstrapped_latencies(
            find_onset, scores, times, n_iters=N_ITERS, random_state=RANDOM_STATE)
        results[f'{label}_boot_peak']= get_bootstrapped_latencies(
            find_peak, scores, times, n_iters=N_ITERS, random_state=RANDOM_STATE)

    results['timepoint'] = times
    out_path = os.path.join(result_dir, 'timing_latency', 
                            f'timing_latency-category.{category}.mat')
    savemat(out_path, results)



def get_bootstrapped_latencies(find_func, scores, times, n_iters=1000, random_state=None,  **kwargs):
    if random_state:
        np.random.seed(random_state)

    n_sub = scores.shape[0]
    timepoints = np.full(n_iters, np.nan)

    for ii in range(n_iters):
        ind = np.random.choice(n_sub, n_sub)
        timepoints[ii] = find_func(
            scores[ind], times, **kwargs)

    return timepoints


def find_onset(scores, times, t_range=T_RANGE, p=.01, n=3):
    '''find the onset time when p is smaller than a given threshold'''

    tslice = slice(np.where(times == t_range[0])[0][0],
                   np.where(times == t_range[1])[0][0])

    scores = scores.copy() - .5  # chance
    pvals = np.array(
        [wilcoxon(s, alternative='greater').pvalue for s in scores[:, tslice].T])

    mask = pvals < p
    id_ = find_first_consecutive_ind(mask, n)
    return times[tslice][id_] if id_ else None


def find_first_consecutive_ind(arr, n):
    count = 0
    for ii, item in enumerate(arr):
        count = count + 1 if item else 0
        if count >= n:
            return ii - n
    return None


def find_peak(scores, times, t_range=T_RANGE, mpd=50):
    '''find the onset time when p is smaller than a given threshold'''
    tslice = slice(np.where(times == t_range[0])[0][0],
                   np.where(times == t_range[1])[0][0])

    score = scores[:, tslice].mean(0)
    peaks = detect_peaks(score, mpd=mpd)

    try:
        return times[tslice][peaks[0]]
    except IndexError:
        return np.nan
