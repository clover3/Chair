from typing import List, Dict, Tuple

from arg.perspectives import es_helper
from arg.perspectives.load import get_claims_from_ids, load_dev_claim_ids
from cache import save_to_pickle
from list_lib import lmap


def save_dev_candidate():
    d_ids: List[int] = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)
    candidates: List[Tuple[Dict, List[Dict]]] = get_all_candidate(claims)
    save_to_pickle(candidates, "pc_dev_candidate")


def get_all_candidate(claims):
    def get_candidate(c: Dict)-> List[Dict]:
        cid = c["cId"]
        claim_text = c["text"]
        lucene_results = es_helper.get_perspective_from_pool(claim_text, 50)
        candidate_list = []
        for rank, (_text, _pid, _score) in enumerate(lucene_results):
            p_entry = {
                'cid': cid,
                'pid': _pid,
                'claim_text': claim_text,
                'perspective_text': _text,
                'score': _score,
            }
            candidate_list.append(p_entry)
        return candidate_list

    candidate = lmap(get_candidate, claims)

    return list(zip(claims, candidate))


if __name__ == "__main__":
    save_dev_candidate()