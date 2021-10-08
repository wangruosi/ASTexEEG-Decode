
from itertools import product

from .eeg import Analysis
from .configs import SUBJECT_IDS, CATEGORIES, STIM_TYPES

# ---------- RUN PREPROCESSING FOR ALL PARTICIPANTS ---------- #


# def run_preprocessing():
#     preproc = Preprocess(subject_ids)
#     preproc.run('raw', save=True)
#     preproc.run('ica', save=True)
#     preproc.run('epochs', preproc='ica_ar', save=True)

# ---------- RUN ANALYSES FOR ALL PARTICIPANTS ---------- #


def run_category_decoding(cross=False):
    ana = Analysis(SUBJECT_IDS)
    for category, stim_type in product(CATEGORIES, STIM_TYPES):
        ana.run('decode_category', stim_type=stim_type,
                category=category, cross=cross)

    # for category in categories:
    # run_get_latency(category)

def run_eeg_rdm():
    ana = Analysis(SUBJECT_IDS)
    ana.run('rdm_eeg', group_averaging=True, stim_type='all')


def run_pairwise_decoding(cross=False):
    ana = Analysis(SUBJECT_IDS)
    for stim_type in STIM_TYPES:
        ana.run('decode_pair', stim_type=stim_type, cross=cross)


def run_tripartite_size_decoding():
    ana = Analysis(SUBJECT_IDS)
    subtypes = ['animal', 'object']
    for subtype, stim_type in product(subtypes, STIM_TYPES):
        ana.run('decode_category',
                stim_type=f'{stim_type}_{subtype}', category='size')


def run_qa_n_trials():
    ana = Analysis(SUBJECT_IDS)
    ana.run('qa_n_trials', by='stimulus')
    ana.run('qa_n_trials', by='condition')


def run_get_latency():
    from .stats.timing import get_latencies
    categories = ('animacy', 'size')
    for category in categories:
        get_latencies(category)