from arg.perspectives.n_gram_feature_collector import PCNgramWorker
from data_generator.job_runner import JobRunner, sydney_working_dir


if __name__ == "__main__":
    def worker_factory(out_dir):
        return PCNgramWorker(
            input_job_name="perspective_paragraph_feature",
            max_para=30,
            out_dir=out_dir
            )


    runner = JobRunner(sydney_working_dir, 606, "pc_ngram_all_train", worker_factory)
    runner.start()


