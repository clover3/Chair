from arg.perspectives.n_gram_feature_collector_subword import PCNgramSubwordWorker
from data_generator.job_runner import JobRunner, sydney_working_dir


if __name__ == "__main__":
    def worker_factory(out_dir):
        return PCNgramSubwordWorker(
            input_job_name="rel_score_to_para_train",
            max_para=30,
            out_dir=out_dir
            )

    runner = JobRunner(sydney_working_dir, 606, "pc_ngram_subword_all_train", worker_factory)
    runner.start()


