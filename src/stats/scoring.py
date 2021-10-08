
import numpy as np
from scipy.stats import wilcoxon
from mne.stats import fdr_correction

from ..configs import T_RANGE

def get_sig_timepoints(xs, times, alpha=.05, t_range=T_RANGE, alternative='two-sided'):
    if t_range is not None:
        tslice = slice(np.where(times == t_range[0])[0][0],
                       np.where(times == t_range[1])[0][0])
    else:
        tslice = slice(0, len(times))

    pvals = np.array(
        [wilcoxon(x, alternative=alternative).pvalue for x in xs[:, tslice].T])
    mask, _ = fdr_correction(pvals, alpha=alpha)

    return times[tslice][mask]
