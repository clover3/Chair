from typing import List, Dict

from arg.pers_evidence.common import get_qck_queries
from arg.pers_evidence.es_helper import get_evidence_from_pool
from arg.perspectives.load import splits
from arg.qck.decl import QCKCandidate, QCKQuery, QCKCandidateWToken
from arg.qck.qcknc_datagen import QCKCandidateI
from cache import save_to_pickle, load_from_pickle
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lmap
from misc_lib import enum_passage


def get_candidate(split) -> Dict[str, List[QCKCandidateI]]:
    tokenizer = get_tokenizer()
    queries = get_qck_queries(split)
    max_seq_length = 512

    def get_candidate_for_query(query: QCKQuery):
        res = get_evidence_from_pool(query.text, 60)
        query_len = len(tokenizer.tokenize(query.text))
        candidate_max_len = max_seq_length - 2 - query_len
        assert candidate_max_len > 100

        output = []
        for text, e_id, score in res:
            tokens = tokenizer.tokenize(text)
            for passage in enum_passage(tokens, candidate_max_len):
                c = QCKCandidateWToken(e_id, "", passage)
                output.append(c)
        return output

    qid_list = lmap(lambda q: q.query_id, queries)
    candidate_list_list = lmap(get_candidate_for_query, queries)
    return dict(zip(qid_list, candidate_list_list))


def get_candidate_w_score(split, score_cut) -> Dict[str, List[QCKCandidate]]:
    queries = get_qck_queries(split)

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


def load_candidate(split) -> Dict[str, List[QCKCandidateI]]:
    return load_from_pickle("pc_evidence_candidate_{}".format(split))


def main():
    for split in splits:
        c = get_candidate(split)
        save_to_pickle(c, "pc_evidence_candidate_{}".format(split))


if __name__ == "__main__":
    main()

