import os
import pickle

import spacy

from data_generator.job_runner import WorkerInterface, JobRunner
from dataset_specific.msmarco.common import load_queries
from epath import job_man_dir
from misc_lib import TEL


class QueryParseWorker(WorkerInterface):
    def __init__(self, queries, n_per_job, out_dir):
        self.out_dir = out_dir
        self.queries = queries
        self.n_per_job = n_per_job
        self.nlp = spacy.load("en_core_web_sm")

    def work(self, job_id):
        st = self.n_per_job * job_id
        ed = self.n_per_job * (job_id+1)
        queries = self.queries[st:ed]
        output = []
        for qid, q_str in TEL(queries):
            q_str = q_str.strip()
            spacy_tokens = self.nlp(q_str)
            out_e = qid, q_str, spacy_tokens
            output.append(out_e)

        save_path = os.path.join(self.out_dir, str(job_id))
        pickle.dump(output, open(save_path, "wb"))



def run_query_parse_jobs(split):
    queries = load_queries(split)
    n_per_job = 50 * 1000
    num_jobs = int(len(queries) / n_per_job) + 1
    print("{} jobs".format(num_jobs))

    def factory(out_dir):
        return QueryParseWorker(queries, n_per_job, out_dir)

    runner = JobRunner(job_man_dir, num_jobs, "msmarco_spacy_query_parse_{}".format(split), factory)
    runner.start()


def main():
    split = "train"
    run_query_parse_jobs(split)


if __name__ == "__main__":
    main()