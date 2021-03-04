
from data_generator.job_runner import JobRunner
from dataset_specific.msmarco.common import load_query_group, top100_doc_ids
from dataset_specific.msmarco.tokenize_jobs.run_tokenize import DummyWorker
from epath import job_man_dir

if __name__ == "__main__":
    split = "dev"
    query_group = load_query_group(split)
    candidate_docs = top100_doc_ids(split)

    def factory(out_dir):
        return DummyWorker(split, query_group, candidate_docs, out_dir)

    runner = JobRunner(job_man_dir, len(query_group)-1, "MSMARCO_{}_tokens_debug".format(split), factory)
    runner.start()
