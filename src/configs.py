'''
Configuration parameters
'''
import os
import os.path as op
from itertools import product

import numpy as np

#######################################################################

# Paths and Files
root_dir = op.abspath(op.join(op.dirname(__file__), '..'))
data_dir = op.join(root_dir, 'data')
eeg_dir = op.join(data_dir, 'preprocessed')            # mne format .fif files
misc_dir = op.join(data_dir, 'misc')
interim_dir = op.join(data_dir, 'interim')          # interim results


# make dir if it does not exists
[os.mkdir(eeg_dir) if not op.isdir(eeg_dir) else None]
[os.mkdir(interim_dir) if not op.isdir(interim_dir) else None]

#######################################################################

# Experiment Designs
stim_types = ('original', 'texform')
levels = dict(size=('big', 'small'),
              animacy=('animal', 'object'))
categories = list(levels.keys())
conditions = tuple(map(lambda x: '/'.join(x),
                       product(stim_types, *levels.values())))

n_stim_types = len(stim_types)
n_trials_per_run = 240
n_runs_per_stim_type = 6
n_runs_total = n_runs_per_stim_type * n_stim_types
n_trials_per_stim_type = n_trials_per_run * n_runs_per_stim_type
n_trials_total = n_trials_per_run * n_runs_total

n_exemplars = 15
n_stimulus = n_exemplars * len(conditions)
trigger_ids = {f'Stimulus/S{i+1:3d}': i+1 for i in range(n_stimulus)}
event_ids = {f'{condition}/object{i+1:03d}': i+1 for i, condition in
             enumerate(np.repeat(conditions, n_exemplars))}

# subject ids
subject_ids = SUBJECT_IDS = list(range(1, 13)) + list(range(14, 20))
n_subs = len(subject_ids)

# preprocessing settings
TMIN, TMAX = -.1, .9    # tmin, tmax for epoching
T_RANGE = (.1, .5)      # time range for statistical testing
LP, HP = .01, 100       # lower and upper band
eog = ('HEOG',)         # EOG channels

# decoding settings
PREPROC = 'ica_ar'      # preprocessing steps
CLF_NAME = 'lda'        # classifer name
N_SUPER = 6             # numbet of super trials

RANDOM_STATE = 314
N_JOBS = 6
