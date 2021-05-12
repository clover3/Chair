from data_generator.job_runner import JobRunner
from dataset_specific.msmarco.common import load_query_group, load_candidate_doc_list_10, load_candidate_doc_top50
from dataset_specific.msmarco.tokenize_for_bm25 import TokenizeForBM25Worker
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS


def run_tokenize_jobs_for_train_split(split):
    query_group = load_query_group(split)
    candidate_docs = load_candidate_doc_list_10(split)

    def factory(out_dir):
        return TokenizeForBM25Worker(split, query_group, candidate_docs, out_dir)

    runner = JobRunnerS(job_man_dir, len(query_group),
                        "MSMARCO_{}_bm25_tokenize".format(split),
                        factory)
    runner.start()


if __name__ == "__main__":
    split = "train"
    run_tokenize_jobs_for_train_split(split)
