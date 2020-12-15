from typing import List, Dict

from arg.pers_evidence.es_helper import get_evidence_from_pool
from arg.pers_evidence.qc_gen import get_qck_queries
from arg.perspectives.load import splits
from arg.qck.decl import QCKCandidate, QCKQuery
from cache import save_to_pickle, load_from_pickle
from list_lib import lmap


def get_candidate(split) -> Dict[str, List[QCKCandidate]]:
    queries = get_qck_queries(split)[:100]

    def get_candidate_for_query(query: QCKQuery):
        res = get_evidence_from_pool(query.text, 10)
        output = []
        for text, e_id, score in res:
            c = QCKCandidate(e_id, text)
            output.append(c)
        return output

    qid_list = lmap(lambda q: q.query_id, queries)
    candidate_list_list = lmap(get_candidate_for_query, queries)
    return dict(zip(qid_list, candidate_list_list))


def get_candidate_w_score(split, score_cut) -> Dict[str, List[QCKCandidate]]:
    queries = get_qck_queries(split)[:100]

    def get_candidate_for_query(query: QCKQuery):
        res = get_evidence_from_pool(query.text, 10)
        output = []
        for text, e_id, score in res:
            if score > score_cut:
                c = QCKCandidate(e_id, text)
                output.append(c)
        return output

    qid_list = lmap(lambda q: q.query_id, queries)
    candidate_list_list = lmap(get_candidate_for_query, queries)
    return dict(zip(qid_list, candidate_list_list))


def load_candidate(split) -> Dict[str, List[QCKCandidate]]:
    return load_from_pickle("pc_evidence_candidate_{}".format(split))


def main():
    for split in splits:
        c = get_candidate(split)
        save_to_pickle(c, "pc_evidence_candidate_{}".format(split))


if __name__ == "__main__":
    main()

