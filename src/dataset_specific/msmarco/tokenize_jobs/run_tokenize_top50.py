from dataset_specific.msmarco.common import load_query_group, load_candidate_doc_list_10, load_candidate_doc_top50
from dataset_specific.msmarco.tokenize_worker import TokenizeWorker
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS

if __name__ == "__main__":
    split = "train"
    query_group = load_query_group(split)
    candidate_docs = load_candidate_doc_top50(split)

    def factory(out_dir):
        return TokenizeWorker(split, query_group, candidate_docs, out_dir)

    runner = JobRunnerS(job_man_dir, len(query_group)-1, "MSMARCO_{}_top50_tokens".format(split), factory)
    runner.start()
