from dataset_specific.msmarco.common import load_query_group
from dataset_specific.msmarco.tokenize_worker_w_nltk import SentLevelTokenizeWorker2
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS


def run_tokenize_jobs_for_train_split(split):
    query_group = load_query_group(split)

    def factory(out_dir):
        return SentLevelTokenizeWorker2(split, query_group, out_dir)

    runner = JobRunnerS(job_man_dir, len(query_group),
                        "MSMARCO_{}_sent_tokens_all".format(split),
                        factory)
    runner.start()


def run_tokenize_jobs_for_train_split_10docs(split):
    query_group = load_query_group(split)


    def factory(out_dir):
        return SentLevelTokenizeWorker2(split, query_group, out_dir)

    runner = JobRunnerS(job_man_dir, len(query_group),
                        "MSMARCO_{}_sent_tokens_10docs".format(split),
                        factory)
    runner.start()


if __name__ == "__main__":
    split = "train"
    run_tokenize_jobs_for_train_split(split)
