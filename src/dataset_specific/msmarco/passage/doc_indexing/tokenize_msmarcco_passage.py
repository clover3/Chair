from collections import Counter

import nltk
from typing import List, Iterable, Callable, Dict, Tuple, Set
import itertools
from cpath import output_path
from data_generator.job_runner import WorkerInterface
from dataset_specific.msmarco.passage_common import enum_passage_corpus
from job_manager.job_runner_with_server import JobRunnerS
from misc_lib import TELI, path_join


class CorpusTokenizeWorker(WorkerInterface):
    def __init__(self, doc_per_job, collection_size, work_dir):
        self.doc_per_job = doc_per_job
        self.collection_size = collection_size
        self.work_dir = work_dir

    def work(self, job_id):
        st = job_id * self.doc_per_job
        ed = st + self.doc_per_job
        save_path = path_join(self.work_dir, str(job_id))
        f = open(save_path, "w", encoding="utf-8")
        itr = itertools.islice(enum_passage_corpus(), st, ed)
        collection_size = self.doc_per_job
        for doc_id, doc_text in TELI(itr, collection_size):
            word_tokens = nltk.word_tokenize(doc_text)
            out_s = " ".join(word_tokens)
            f.write("{}\t{}\n".format(doc_id, out_s))
        f.close()


def main():
    doc_per_job = 1000000
    collection_size = 8841823
    num_jobs = int(collection_size / doc_per_job) + 1
    work_path = path_join(output_path, "msmarco")
    job_name = "msmarco_passage_tokenize"

    def factory(work_path):
        corpus_tokenizer = CorpusTokenizeWorker(doc_per_job, collection_size, work_path)
        return corpus_tokenizer

    runner = JobRunnerS(work_path, num_jobs, job_name, factory)
    runner.start()


if __name__ == "__main__":
    main()