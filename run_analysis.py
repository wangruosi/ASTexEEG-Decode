
from itertools import product

from src.eeg import Analysis
from src.configs import subject_ids, categories, stim_types

# ---------- RUN PREPROCESSING FOR ALL PARTICIPANTS ---------- #


# def run_preprocessing():
#     preproc = Preprocess(subject_ids)
#     preproc.run('raw', save=True)
#     preproc.run('ica', save=True)
#     preproc.run('epochs', preproc='ica_ar', save=True)

# ---------- RUN ANALYSES FOR ALL PARTICIPANTS ---------- #


def run_category_decoding():
    ana = Analysis(subject_ids)
    cross_choices = [False, True]
    for cross, category, stim_type in product(cross_choices, categories, stim_types):
        ana.run('decode_category', stim_type=stim_type,
                category=category, cross=cross)

    # rdm mds visualization
    ana.run('rdm_eeg')

    # for category in categories:
    # run_get_latency(category)


def run_pairwise_decoding():
    ana = Analysis(subject_ids)
    cross_choices = [False, True]
    for cross, stim_type in product(cross_choices, stim_types):
        ana.run('decode_pair', stim_type=stim_type, cross=cross)


def run_tripartite_size_decoding():
    ana = Analysis(subject_ids)
    subtypes = ['animal', 'object']
    for subtype, stim_type in product(subtypes, stim_types):
        ana.run('decode_category',
                stim_type=f'{stim_type}_{subtype}', category='size')


def run_qa_n_trials():
    ana = Analysis(subject_ids)
    ana.run('qa_n_trials', by='stimulus')
    ana.run('qa_n_trials', by='condition')
