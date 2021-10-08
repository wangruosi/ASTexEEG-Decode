
from src import run

if __name__ == '__main__':

    # Count the number of trials that survive artifact rejection
    run.run_qa_n_trials()
    
    # Decode animacy (animal vs. object) and size (big vs. small)
    run.run_category_decoding(cross=False)
    run.run_category_decoding(cross=True)
    
    # Estimate the onset/peak latencies via bootstrapping
    run.run_get_latency()

    # Make EEG representational dissimilarity matrices
    run.run_eeg_rdm()
    
    # Decode size separately for animals and objects
    run.run_tripartite_size_decoding()
    
    # Decode all pairs of objects
    run.run_pairwise_decoding()
    