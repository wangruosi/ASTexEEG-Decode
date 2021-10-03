import numpy as np
import scipy

from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import _cov


class SuperTrial:

    def __init__(self, epochs, n_super):
        """
        Parameters
        ----------
        data: np.ndarray()
            EEG data (n_trials, n_ch, n_times)
        labels : List<int>, np.ndarray()
            Event labels for each trial
        n_super : int
            The number of super-trials to create
        """
        data = epochs.get_data()
        labels = epochs.events[:, 2]
        times = epochs.times

        self.data = data
        self.labels = labels
        self.times = times
        self.n_super = n_super
        self.classes = np.unique(self.labels)

    def split(self):
        indices = []
        for c in self.classes:
            arr = np.where(self.labels == c)[0]
            np.random.shuffle(arr)
            n_split = self.n_super if self.n_super else len(arr)
            indices.extend(np.array_split(arr, n_split))
        self.super_indices = indices
        return self

        # self.super_indices = [kf.]

    def average(self):
        """
        Return
        --------
        super_trial_data: np.ndarray()
            Super-trial data averaged across multiple trials
        super_trial_labels: np.array()
            Label for each super-trial
        """
        super_trial_data = [self.data[ind].mean(
            0)for ind in self.super_indices]
        super_trial_labels = [np.unique(self.labels[ind]).item()
                              for ind in self.super_indices]

        return np.array(super_trial_data), np.array(super_trial_labels)

    def split_average(self):
        return self.split().average()


class MultivariateNoiseNormalizer():
    """
    EEG patterns are normalized by means of a covariance matrix based on the epoch method

    Ref: Guggenmos, Sterzer, Cichy. NeuroImage. (2018)
    """

    def fit(self, X, y):
        """Compute the covariance matrix to be used for later normalisation.
        Parameters
        ----------
        X : np.ndarray() [n_samples, n_channels, n_times]
            The data used to compute
        y : np.array()

        """
        n_times = X.shape[-1]
        classes = np.unique(y)

        sigmas = [np.mean([_cov(X[y == c, :, t], shrinkage='auto')
                           for t in range(n_times)], axis=0)
                  for c in classes]
        sigma = np.mean(sigmas, axis=0)

        sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5)
        self.sigma_inv = sigma_inv
        return self

    def transform(self, X):
        return (X.swapaxes(1, 2) @ self.sigma_inv).swapaxes(1, 2)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class GroupStratifiedKFold():
    """
    Stratified Shuffled Split with non-overlapping groups
    """

    def __init__(self, n_splits=5, mapping=None):
        self.n_splits = n_splits
        if mapping:
            self.mapping = mapping
        else:
            self.mapping = dict(
                zip(np.arange(1, 121), np.repeat(np.arange(1, 9), 15)))

    def split(self, X, y, groups=None):
        # check input
        assert X.shape[0] == len(y) == len(groups)
        ids = np.arange(len(groups))  # all index
        group_X = np.array(sorted(set(groups)))
        group_y = np.array([self.mapping[x] for x in group_X])
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True)
        for group_train, group_test in cv.split(group_X, group_y):
            train = ids[np.isin(groups, group_X[group_train])]
            test = ids[np.isin(groups, group_X[group_test])]
            yield train, test

    def get_n_splits(self):
        return self.n_splits
