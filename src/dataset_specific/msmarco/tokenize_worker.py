import os
import pickle
from typing import List

from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.common import MSMarcoDataReader, MSMarcoDoc, load_per_query_docs
from log_lib import log_variables
from misc_lib import TimeEstimator


class TokenizeWorker:
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
        ticker = TimeEstimator(len(qid_list))
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
            if qid not in self.candidate_docs_d:
                continue

            docs: List[MSMarcoDoc] = load_per_query_docs(qid, empty_doc_fn)
            ticker.tick()

            target_docs = self.candidate_docs_d[qid]
            tokens_d = {}
            for d in docs:
                if d.doc_id in target_docs:
                    text = d.title + " " + d.body
                    tokens = self.tokenizer.tokenize(text)
                    tokens_d[d.doc_id] = tokens

            if len(tokens_d) < len(target_docs):
                log_variables(job_id, qid)
                print("{} of {} not found".format(len(tokens_d), len(target_docs)))

            save_path = os.path.join(self.out_dir, str(qid))
            pickle.dump(tokens_d, open(save_path, "wb"))