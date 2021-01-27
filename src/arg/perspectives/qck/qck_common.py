from typing import List, Dict

from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import load_claim_ids_for_split, get_claims_from_ids
from arg.qck.decl import QCKQuery, QCKCandidate
from arg.qck.get_candidate_from_ranked_list import get_candidate_ids_from_ranked_list_path, \
    get_candidate_ids_from_ranked_list
from list_lib import lmap, dict_value_map


def get_qck_queries(split) -> List[QCKQuery]:
    d_ids: List[int] = list(load_claim_ids_for_split(split))
    return get_qck_queries_from_cids(d_ids)


def get_qck_queries_from_cids(d_ids: List[int]):
    claims: List[Dict] = get_claims_from_ids(d_ids)

    def claim_to_query(claim: Dict):
        return QCKQuery(str(claim['cId']), claim['text'])

    queries: List[QCKQuery] = lmap(claim_to_query, claims)
    return queries


def get_qck_candidate_from_candidate_id(candidate_id: str):
    text = perspective_getter(int(candidate_id))
    return QCKCandidate(candidate_id, text)


def doc_id_to_candidate(doc_id: str) -> QCKCandidate:
    return QCKCandidate(doc_id, perspective_getter(int(doc_id)))


def add_texts(doc_ids: List[str]):
    return lmap(doc_id_to_candidate, doc_ids)


def get_qck_candidate_from_ranked_list_path(ranked_list_path) -> Dict[str, List[QCKCandidate]]:
    d: Dict[str, List[str]] = get_candidate_ids_from_ranked_list_path(ranked_list_path)
    return dict_value_map(add_texts, d)


def get_qck_candidate_from_ranked_list(ranked_list) -> Dict[str, List[QCKCandidate]]:
    d: Dict[str, List[str]] = get_candidate_ids_from_ranked_list(ranked_list)
    return dict_value_map(add_texts, d)
