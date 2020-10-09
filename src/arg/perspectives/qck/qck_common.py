from typing import List, Dict

from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import load_claim_ids_for_split, get_claims_from_ids
from arg.qck.decl import QCKQuery, QCKCandidate
from list_lib import lmap


def get_qck_queries(split):
    d_ids = list(load_claim_ids_for_split(split))
    claims: List[Dict] = get_claims_from_ids(d_ids)

    def claim_to_query(claim: Dict):
        return QCKQuery(str(claim['cId']), claim['text'])

    queries: List[QCKQuery] = lmap(claim_to_query, claims)
    return queries


def get_qck_candidate_from_candidate_id(candidate_id: str):
    text = perspective_getter(int(candidate_id))
    return QCKCandidate(candidate_id, text)
