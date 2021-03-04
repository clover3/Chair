from typing import List

from data_generator.job_runner import JobRunner
from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.common import load_per_query_docs, \
    load_query_group, MSMarcoDoc, load_candidate_doc_list_1, MSMarcoDataReader
from dataset_specific.msmarco.tokenize_worker import TokenizeWorker
from epath import job_man_dir
from log_lib import log_variables


class DummyWorker:
    def __init__(self,
                 split,
                 query_group,
                 candidate_docs_d, out_dir):
        self.query_group = query_group
        self.tokenizer = get_tokenizer()
        self.candidate_docs_d = candidate_docs_d
        self.out_dir = out_dir
        self.ms_reader = MSMarcoDataReader(split)

    def work(self, job_id):
        qid_list = self.query_group[job_id]
        missing_rel_cnt = 0
        missing_nrel_cnt = 0
        def empty_doc_fn(query_id, doc_id):
            rel_docs = self.ms_reader.qrel[query_id]
            nonlocal missing_rel_cnt
            nonlocal missing_nrel_cnt
            if doc_id in rel_docs:
                missing_rel_cnt += 1
            else:
                missing_nrel_cnt += 1

        for qid in qid_list:
            docs: List[MSMarcoDoc] = load_per_query_docs(qid, empty_doc_fn)
            if qid not in self.candidate_docs_d:
                continue

            target_docs = self.candidate_docs_d[qid]
            tokens_d = {}
            for d in docs:
                if d.doc_id in target_docs:
                    tokens_d[d.doc_id] = []

            if len(tokens_d) < len(target_docs):
                log_variables(job_id, qid, tokens_d, target_docs)
                not_found_docs = list([doc_id for doc_id in target_docs if doc_id not in tokens_d])
                print("{} of {} not found: {}".format(len(not_found_docs),
                                                      len(target_docs), not_found_docs))


if __name__ == "__main__":
    split = "train"
    query_group = load_query_group(split)
    candidate_docs = load_candidate_doc_list_1(split)

    def factory(out_dir):
        return TokenizeWorker(split, query_group, candidate_docs, out_dir)

    runner = JobRunner(job_man_dir, len(query_group)-1, "MSMARCO_{}_tokens".format(split), factory)
    runner.start()
