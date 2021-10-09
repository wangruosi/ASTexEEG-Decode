
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


def run_category_decoding(sids=SUBJECT_IDS, cross=False):
    ana = Analysis(sids)
    for category, stim_type in product(CATEGORIES, STIM_TYPES):
        ana.run('decode_category', stim_type=stim_type,
                category=category, cross=cross)

    # for category in categories:
    # run_get_latency(category)

def run_eeg_rdm(sids=SUBJECT_IDS):
    ana = Analysis(sids)
    ana.run('rdm_eeg', group_averaging=True, stim_type='all')


def run_pairwise_decoding(sids=SUBJECT_IDS, cross=False):
    ana = Analysis(sids)
    for stim_type in STIM_TYPES:
        ana.run('decode_pair', stim_type=stim_type, cross=cross)


def run_tripartite_size_decoding(sids=SUBJECT_IDS, cross=False):
    ana = Analysis(sids)
    subtypes = ['animal', 'object']
    for subtype, stim_type in product(subtypes, STIM_TYPES):
        ana.run('decode_category',
                stim_type=f'{stim_type}_{subtype}', category='size', cross=cross)


def run_qa_n_trials(sids=SUBJECT_IDS):
    ana = Analysis(sids)
    ana.run('qa_n_trials', by='stimulus')
    ana.run('qa_n_trials', by='condition')


def run_get_latency():
    from .stats.timing import get_latencies
    categories = ('animacy', 'size')
    for category in categories:
        get_latencies(category)