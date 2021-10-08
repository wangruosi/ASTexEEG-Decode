

from ..configs import CONDITIONS
from .data import SubjectModel


def count_num_of_trials(subject_id, by='stimulus'):
    '''
    Count the num trials for each condition
    '''
    params = locals()

    # --------------- LOAD --------------- #
    subj = SubjectModel(subject_id)
    epochs = subj.load('epochs')

    if by == 'stimulus':
        labels = list(epochs.event_id.keys())
    elif by == 'condition':
        labels = CONDITIONS
    else:
        raise ValueError

    n_trials = [len(epochs[l]) for l in labels]

    return {
        'info': dict(label=labels),
        'params': params,
        'n_trials': n_trials
    }
