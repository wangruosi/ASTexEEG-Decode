
import numpy as np
from scipy.spatial.distance import pdist, squareform

from .data import SubjectModel
from .utils import MultivariateNoiseNormalizer

N_ITERS = 20
PREPROC = 'ica_ar'


def compute_eeg_rdm(subject_id,
                    stim_type='all',
                    metric='euclid',
                    whiten=True):
    '''compute rdm for an individual subject'''
    params = locals()

    # --------------- LOAD --------------- #
    subj = SubjectModel(subject_id)
    epochs = subj.load('epochs', preproc=PREPROC)
    epochs = epochs if stim_type == 'all' else epochs[stim_type]

    X, labels = epochs.get_data(), epochs.events[:, 2]
    rdms = get_noncv_dist(
        X, labels, metric=metric, whiten=whiten)
    # --------------- SAVE --------------- #
    # specify variable info and make sure it matches the result
    var_info = dict(stimulus=np.unique(labels),
                    timepoint=epochs.times)
    return {
        'info': var_info,
        'params': params,
        'rdm': rdms,
    }


def get_noncv_dist(X, labels, metric, whiten):
    """
    Non-cross-validated distance mesasure 
    """
    n_times = X.shape[-1]
    classes = np.unique(labels)

    if whiten:
        # 2. Whitening using the Epoch method
        mnn = MultivariateNoiseNormalizer()
        X = mnn.fit_transform(X, labels)

    # 3. Apply distance mesasure
    # reshaped = X.reshape(n_classes, -1, n_ch, n_times).mean(1)
    averaged = np.array([X[labels == c, :, :].mean(0) for c in classes])
    return np.array([squareform(pdist(averaged[:, :, t], metric=metric))
                     for t in range(n_times)])
