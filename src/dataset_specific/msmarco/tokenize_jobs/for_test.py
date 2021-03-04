from dataset_specific.msmarco.tokenize_jobs.run_tokenize_dev import run_tokenize_jobs_for_prediction_split

if __name__ == "__main__":
    split = "test"
    run_tokenize_jobs_for_prediction_split(split)
