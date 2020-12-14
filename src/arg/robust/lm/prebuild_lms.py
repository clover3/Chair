import os
import pickle
from collections import Counter
from typing import List, Dict

from arg.perspectives.pc_tokenizer import PCTokenizer
from data_generator.data_parser import trec
from data_generator.data_parser.robust2 import load_robust_qrel, load_bm25_best
from data_generator.job_runner import JobRunner
from epath import job_man_dir
from misc_lib import TimeEstimator, tprint

n_jobs = 5

def all_doc_ids_of_interest() -> List[str]:
    qrel = load_robust_qrel()
    all_doc_id_set = set()
    for query in qrel.keys():
        judgement = qrel[query]
        for doc_id, score in judgement.items():
            all_doc_id_set.add(doc_id)

    top_k = 1000
    galago_rank = load_bm25_best()
    for query_id, ranked_list in galago_rank.items():
        ranked_list.sort(key=lambda x:x[1])
        all_doc_id_set.update([x[0] for x in ranked_list[:top_k]])

    all_doc_id_list = list(all_doc_id_set)
    all_doc_id_list.sort()

    return all_doc_id_list


class Worker:
    def __init__(self, out_dir):
        robust_path = "/mnt/nfs/work3/youngwookim/data/robust04"
        tprint("Loading doc ids")
        self.doc_ids = all_doc_ids_of_interest()
        tprint("Loading robust docs")
        self.docs: Dict[str, str] = trec.load_robust(robust_path)
        tprint("Start processing")

        n_docs = len(self.doc_ids)
        docs_per_job = int((n_docs+n_jobs) / 5)
        self.docs_per_job = docs_per_job
        self.tokenizer = PCTokenizer()
        self.out_dir = out_dir

    def work(self, job_id):
        doc_id_to_count = dict()
        st = job_id * self.docs_per_job
        ed = st + self.docs_per_job
        todo = self.doc_ids[st:ed]
        ticker = TimeEstimator(len(todo))
        for doc_id in todo:
            try:
                text = self.docs[doc_id]
                tokens = self.tokenizer.tokenize_stem(text)
                counter = Counter(tokens)
                doc_id_to_count[doc_id] = counter
                ticker.tick()
            except KeyError as e:
                print(e)
                print("key error")
                pass

        save_path = os.path.join(self.out_dir, str(job_id))
        pickle.dump(doc_id_to_count, open(save_path, "wb"))


if __name__ == "__main__":
    runner = JobRunner(job_man_dir, n_jobs-1, "robust_tokens", Worker)
    runner.start()

