'''
Classes to centralize data loading
'''
import os
import os.path as op
from scipy.io import loadmat, savemat
import mne

from ..configs import eeg_dir, result_dir


class SubjectModel():

    def __init__(self, subject_id):
        self.subject = f'ASTexEEG_Exp01Sub{subject_id:02d}'
        self._epo_fname = f'{self.subject}-filt-ica_ar-epo.fif.gz'

    def load(self, file_type='epochs'):
        if file_type == 'epochs':
            in_path = op.join(eeg_dir, self._epo_fname)
            return mne.read_epochs(in_path)
        else:
            raise ValueError(f'{file_type} is not an option')


class GroupModel():
    analyses = {'decode_category': ['stim_type', 'category', 'cross'],
                'decode_pair': ['stim_type', 'cross'],
                'qa_n_trials': ['by'],
                'rdm_eeg': ['stim_type'],
                'timing_latency': ['category']}

    def __init__(self):
        for ana in GroupModel.analyses:
            ana_dir = op.join(result_dir, ana)
            if not op.isdir(ana_dir):
                os.mkdir(ana_dir)

    def load(self, analysis, **kwargs):
        assert(analysis in GroupModel.analyses)

        params = {k: kwargs[k] for k in GroupModel.analyses[analysis]}
        label = "-".join([f'{k}.{v}' for k, v in params.items()])
        load_file = op.join(result_dir, analysis,
                            f'{analysis}-{label}.mat')
        return loadmat(load_file)

    def save(self, analysis, data_to_save, params_label):
        assert(analysis in GroupModel.analyses)
        save_file = op.join(result_dir, analysis,
                            f'{analysis}-{params_label}.mat')
        savemat(save_file, data_to_save, do_compression=True)
