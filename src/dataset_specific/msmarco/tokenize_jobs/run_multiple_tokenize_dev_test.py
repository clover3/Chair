from data_generator.job_runner import JobRunner
from dataset_specific.msmarco.common import load_query_group, top100_doc_ids
from dataset_specific.msmarco.multiple_tokenize_worker import MultipleTokenizeWorker
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS


def run_tokenize_jobs_for_pred_split(split):
    query_group = load_query_group(split)
    candidate_docs = top100_doc_ids(split)
    max_sent_length = 64 * 5
    max_title_length = 64 * 5

    def factory(out_dir):
        return MultipleTokenizeWorker(split, query_group, candidate_docs, max_sent_length, max_title_length, out_dir)

    runner = JobRunnerS(job_man_dir, len(query_group),
                        "MSMARCO_{}_multiple_tokenize".format(split),
                        factory)
    runner.start()


if __name__ == "__main__":
    run_tokenize_jobs_for_pred_split("dev")
    run_tokenize_jobs_for_pred_split("test")
