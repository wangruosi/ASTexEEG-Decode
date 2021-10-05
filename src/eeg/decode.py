from itertools import combinations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from mne.decoding import (GeneralizingEstimator,
                          SlidingEstimator, cross_val_multiscore)


from ..configs import (RANDOM_STATE, N_SUPER, CLF_NAME, PREPROC)
from .data import SubjectModel
from .utils import (SuperTrial, GroupStratifiedKFold)


N_ITERS = 20


def decode_category(subject_id,
                    stim_type,
                    category,
                    cross=False):
    ''' run category decoding for a subject'''
    # save input variables
    params = locals()
    stim_type = stim_type.replace('_', '/')

    # set random state to make result reproducible
    np.random.seed(RANDOM_STATE)

    # exemplar based splits
    n_splits = 5
    gkf = GroupStratifiedKFold(n_splits=n_splits)
    # make a time-resolved estimator with the selected classifier
    est = make_estimator('sliding', CLF_NAME)

    # --------------- LOAD --------------- #
    # load eeg epochs for this subject
    subm = SubjectModel(subject_id)
    epochs = subm.load('epochs')

    # create a mapping between stimulus id and the examined category
    category_mapping = get_category_mapping(epochs.event_id, category)

    if cross:
        cross_stim_type = _get_cross_stim_type(stim_type)
        cross_spt = SuperTrial(epochs[cross_stim_type], N_SUPER)

    spt = SuperTrial(epochs[stim_type], N_SUPER)

    n_times = len(spt.times)
    n_total_super = len(spt.classes) * N_SUPER
    n_components = len(set(category_mapping.values())) - 1
    projections = np.full(
        (N_ITERS, n_total_super, n_times, n_components), np.nan)

    scores = []
    for i_iter in range(N_ITERS):

        # Compute super-trials
        X, labels = spt.split_average()
        y = np.array([category_mapping.get(l) for l in labels])

        if cross:
            cross_X, _ = cross_spt.split_average()

        for train, test in gkf.split(X, y, groups=labels):

            X_train, y_train, y_test = X[train], y[train], y[test]
            X_test = cross_X[test] if cross else X[test]

            est.fit(X_train, y_train)
            scores.append(est.score(X_test, y_test))
            projections[i_iter, test] = est.transform(X_test)
            # dec_values[i_iter, test] = est.decision_function(X_test)

    return {
        'info': dict(timepoint=spt.times),
        'params': params,
        'score': np.array(scores).mean(0),
        'projection': np.mean(np.reshape(projections,
                                         (N_ITERS, -1, N_SUPER, n_times, n_components)), axis=(0, 2)).squeeze()
    }


def decode_pair(subject_id,
                stim_type,
                cross=False):
    ''' run pairwise decoding for a subject'''
    # save input variables
    params = locals()

    # set random state to make result reproducible
    np.random.seed(RANDOM_STATE)

    # --------------- LOAD --------------- #
    # load eeg epochs for this subject
    subm = SubjectModel(subject_id)
    epochs = subm.load('epochs')
    epochs.decimate(2)

    if cross:
        cross_epochs = epochs[_get_cross_stim_type(stim_type)]
        cross_epochs_data_list = [epochs[id_].get_data()
                                  for id_ in cross_epochs.event_id]
        cross_epochs_counts = list(
            map(lambda x: x.shape[0], cross_epochs_data_list))

    epochs = epochs[stim_type]
    epochs_data_list = [epochs[id_].get_data() for id_ in epochs.event_id]
    epochs_counts = list(map(lambda x: x.shape[0], epochs_data_list))

    n_labels, n_times = len(epochs_data_list), len(epochs.times)
    scores = np.zeros((n_labels, n_labels, n_times))

    est = make_estimator('sliding', CLF_NAME)
    sss = StratifiedShuffleSplit(n_splits=5, test_size=.2)

    for c1, c2 in combinations(range(n_labels), 2):
        X = np.concatenate([epochs_data_list[c1],
                            epochs_data_list[c2]])
        y = np.concatenate([np.ones(epochs_counts[c1]),
                            -np.ones(epochs_counts[c2])])

        if not cross:
            scores[c1, c2] = scores[c2, c1] = np.mean(cross_val_multiscore(
                est, X, y, cv=sss), axis=0)
        else:
            cross_X = np.concatenate([cross_epochs_data_list[c1],
                                      cross_epochs_data_list[c2]])
            cross_y = np.concatenate([np.ones(cross_epochs_counts[c1]),
                                      -np.ones(cross_epochs_counts[c2])])
            scores[c1, c2] = scores[c2, c1] = np.mean(_cross_val_multiscore(
                est, X, y, cross_X, cross_y, cv=sss), axis=0)

    return {
        'info': dict(timepoint=epochs.times),
        'params': params,
        'score': scores,
    }

# make decoding pipeline


def make_estimator(method, clf):

    if clf == 'lda':
        clf_fun = LinearDiscriminantAnalysis()

    elif clf == 'svm':
        clf_fun = SVC(kernel='linear')

    clf_kwargs = dict(scoring="roc_auc", n_jobs=-1)
    if method == 'sliding':
        est = SlidingEstimator(clf_fun, **clf_kwargs)

    elif method == 'generalizing':
        est = GeneralizingEstimator(clf_fun, **clf_kwargs)

    return est


def get_category_mapping(event_id, category):
    if category == 'animacy':
        category_dict = {'animal': 1, 'object': 0}
        def key_maker(x): return x[2]
    elif category == 'size':
        category_dict = {'big': 1, 'small': 0}
        def key_maker(x): return x[1]
    elif category == 'all':
        category_dict = {('big', 'animal'): 1,
                         ('small', 'animal'): 2,
                         ('big', 'object'): 3,
                         ('small', 'object'): 4}

        def key_maker(x): return tuple(x[1:3])

    mapping = dict()
    for label, id_ in event_id.items():
        labels = label.split('/')
        mapping[id_] = category_dict.get(key_maker(labels))

    return mapping


def _get_cross_stim_type(stim_type):
    mapping = {
        'original': 'texform',
        'texform': 'original'
    }
    if '/' in stim_type:
        splits = stim_type.split('/')
        return f'{mapping.get(splits[0])}/{splits[1]}'
    else:
        return mapping.get(stim_type)


def _cross_val_multiscore(est, X, y, cross_X, cross_y, cv):
    scores = []
    for (train, _), (_, cross_test) in zip(cv.split(X=X, y=y), cv.split(X=cross_X, y=cross_y)):
        est.fit(X=X[train], y=y[train])
        scores.append(est.score(X=cross_X[cross_test], y=cross_y[cross_test]))
    return np.array(scores)
