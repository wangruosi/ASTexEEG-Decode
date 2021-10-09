
# from mne.parallel import parallel_func
import os.path as op
import time
from collections import defaultdict
import logging

import numpy as np

from . import decode, distance, quality
from ..configs import result_dir
from .data import GroupModel


logging.basicConfig(filename=op.join(result_dir, 'run.log'), level=logging.INFO,
                    format='%(levelname)s:%(message)s')

# class Preprocess():
#     from src.eeg import preprocessing
#     preproc_funcs = {
#         'raw': preprocessing.make_raw,
#         'ica': preprocessing.make_ica,
#         'epochs': preprocessing.make_epochs
#     }

#     def __init__(self, subject_ids=None):
#         self.subject_ids = subject_ids

#     def run_subject(self, subject_id, preprocess_name, **kwargs):
#         preproc_func = Preprocess.preproc_funcs.get(preprocess_name)
#         preproc_func(subject_id, **kwargs)

#     def run(self, preprocess_name, **kwargs):
#         preproc_func = Preprocess.preproc_funcs.get(preprocess_name)
#         parallel, run_func, _ = parallel_func(preproc_func, n_jobs=N_JOBS)
#         parallel(run_func(id_, **kwargs) for id_ in self.subject_ids)


class Analysis():
    ana_funcs = {
        'decode_category': decode.decode_category,
        'decode_pair': decode.decode_pair,
        'rdm_eeg': distance.compute_eeg_rdm,
        'qa_n_trials': quality.count_num_of_trials
    }
    preproc_default = dict(preproc='')

    def __init__(self, subject_ids):
        self.subject_ids = subject_ids

    def run(self, analysis_name, group_averaging=False, **kwargs):

        tstart = time.time()
        params_label = "-".join([f'{k}.{v}' for k, v in kwargs.items()])
        logging.info(f'{analysis_name}-{params_label} start...')
    
        ana_func = Analysis.ana_funcs.get(analysis_name)
        grpm, results = GroupModel(), defaultdict(list)

        for subject_id in self.subject_ids:
            result = ana_func(subject_id, **kwargs)

            for k, v in result.items():
                if k not in ('info'):
                    results[k].append(v)

        # combine subject results 
        for k, v in results.items():
            results[k] = np.array(v)
            if group_averaging:
                results[k] = results[k].mean(0)

        # unpack info dictionary
        for k, v in result['info'].items():
            results[k] = v

        # add subj
        if not group_averaging:
            results['subject'] = [f'Sub{s:02d}' for s in self.subject_ids]
        
        
        grpm.save(analysis_name, results, params_label)
        logging.info(f'{analysis_name} finished ({(time.time()-tstart)/60:.2f} mins)\n')
