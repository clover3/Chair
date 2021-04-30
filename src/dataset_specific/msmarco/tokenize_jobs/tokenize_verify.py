import os
import sys

from log_lib import log_variables
from misc_lib import TimeEstimator
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource


class Verifier:
    def __init__(self):
        self.resource = ProcessedResource("train")

    def work(self, job_id):
        qid_list = self.resource.query_group[job_id]
        for qid in qid_list:
            if qid not in self.resource.candidate_doc_d:
                continue

            target_docs = self.resource.candidate_doc_d[qid]
            tokens_d = self.resource.get_doc_tokens_d(qid)

            for doc_id in target_docs:
                if doc_id not in tokens_d:
                    log_variables(qid, target_docs)
                    print("Not foudn: ", doc_id)


if __name__ == "__main__":
    v = Verifier()
    v.work(int(sys.argv[1]))