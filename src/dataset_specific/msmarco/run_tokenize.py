import os
import pickle

from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.common import load_per_query_docs, \
    load_queries
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from list_lib import left


class Worker:
    def __init__(self, qids, out_dir):
        self.qids = qids
        self.tokenizer = get_tokenizer()
        self.out_dir = out_dir

    def work(self, job_id):
        query_id = self.qids[job_id]
        docs = load_per_query_docs(query_id)
        tokens_d = {}
        for d in docs:
            text = d.title + " " + d.body
            tokens = self.tokenizer.tokenize(text)
            tokens_d[d.doc_id] = tokens

        save_path = os.path.join(self.out_dir, str(query_id))
        pickle.dump(tokens_d, open(save_path, "wb"))


if __name__ == "__main__":
    for split in ["train", "dev"]:
        qids = left(load_queries(split))

        def factory(out_dir):
            return Worker(qids, out_dir)

        runner = JobRunnerS(job_man_dir, len(qids), "MSMARCO_{}_tokens".format(split), factory)
        runner.start()
