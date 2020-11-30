# use top-k candidate as payload
from typing import List

from arg.qck.decl import QKUnit, QCKCandidate, QCKQuery
from cache import load_from_pickle
from data_generator.data_parser.robust2 import load_qrel


def load_qk_robust_heldout(data_id) -> List[QKUnit]:
    return load_from_pickle("robust_qk_candidate_{}".format(data_id))


class QRel:
    def __init__(self):
        qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"
        self.judgement = load_qrel(qrel_path)

    def is_correct(self, query: QCKQuery, candidate: QCKCandidate):
        qid = query.query_id
        doc_id = candidate.id
        if qid not in self.judgement:
            return 0
        d = self.judgement[qid]
        if doc_id in d:
            return d[doc_id]
        else:
            return 0


