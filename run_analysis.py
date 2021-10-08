
from src import run

if __name__ == '__main__':

    run.run_qa_n_trials()
    run.run_category_decoding(cross=False)
    run.run_category_decoding(cross=True)
    run.run_get_latency()
    run.run_eeg_rdm()
    run.run_tripartite_size_decoding()
    run.run_pairwise_decoding()
    