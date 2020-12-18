from typing import List, Iterable, Dict

from arg.pers_evidence.common import get_qck_queries
from arg.pers_evidence.es_helper import get_evidence_from_pool
from arg.perspectives.load import splits, load_evidence_dict, evidence_gold_dict_str_qid
from arg.qck.decl import QCKCandidate, QCKQuery, QCKCandidateWToken
from arg.qck.qcknc_datagen import QCKCandidateI
from cache import save_to_pickle, load_from_pickle
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lmap, lflatten
from misc_lib import enum_passage


def get_candidate(split) -> Dict[str, List[QCKCandidateI]]:
    tokenizer = get_tokenizer()
    queries = get_qck_queries(split)
    max_seq_length = 512

    def get_candidate_for_query(query: QCKQuery):
        res = get_evidence_from_pool(query.text, 60)
        query_len = len(tokenizer.tokenize(query.text))
        candidate_max_len = max_seq_length - 3 - query_len

        output = []
        for text, e_id, score in res:
            tokens = tokenizer.tokenize(text)
            for passage in enum_passage(tokens, candidate_max_len):
                c = QCKCandidateWToken(str(e_id), "", passage)
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


def load_top_rank_candidate(split) -> Dict[str, List[QCKCandidateI]]:
    return load_from_pickle("pc_evidence_candidate_{}".format(split))


def load_bal_candidate(split) -> Dict[str, List[QCKCandidateI]]:
    return load_from_pickle("pc_evi_ex_candidate_{}_bal".format(split))


def get_ex_candidate_for_training(split, balanced=True, cached=False) -> Dict[str, List[QCKCandidateI]]:
    if cached:
        bow_ranked = load_top_rank_candidate(split)
    else:
        bow_ranked = get_candidate(split)
    tokenizer = get_tokenizer()
    evi_dict: Dict[int, str] = load_evidence_dict()
    evi_gold_dict: Dict[str, List[int]] = evidence_gold_dict_str_qid()
    queries = get_qck_queries(split)
    max_seq_length = 512
    out_d = {}
    for query in queries:
        qid = query.query_id
        c_list = bow_ranked[qid]
        gold_e_ids: List[int] = evi_gold_dict[qid]
        top_ranked: List[int] = lmap(int, map(QCKCandidate.get_id, c_list))
        query_len = len(tokenizer.tokenize(query.text))
        candidate_max_len = max_seq_length - 3 - query_len
        neg_e_ids = []
        for e_id in set(top_ranked):
            if e_id not in gold_e_ids:
                neg_e_ids.append(e_id)
            if balanced and len(neg_e_ids) == len(gold_e_ids):
                break

        def make_candidate(e_id: int) -> Iterable[QCKCandidate]:
            text = evi_dict[e_id]
            tokens = tokenizer.tokenize(text)
            for passage in enum_passage(tokens, candidate_max_len):
                yield QCKCandidateWToken(str(e_id), "", passage)

        new_list = lflatten(map(make_candidate, gold_e_ids + neg_e_ids))
        out_d[qid] = new_list
    return out_d


def save_to_cache():
    for split in splits:
        c = get_candidate(split)
        save_to_pickle(c, "pc_evidence_candidate_{}".format(split))


def main():
    for split in splits:
        c = get_ex_candidate_for_training(split, True)
        save_to_pickle(c, "pc_evi_ex_candidate_{}_bal".format(split))


if __name__ == "__main__":
    main()

