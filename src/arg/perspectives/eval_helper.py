from typing import List, Dict, Tuple

from arg.perspectives import es_helper
from arg.perspectives.load import get_claims_from_ids, load_dev_claim_ids, load_claim_ids_for_split, splits
from arg.perspectives.pc_tokenizer import PCTokenizer
from arg.perspectives.types import CPIDPair
from cache import save_to_pickle, load_from_pickle
from list_lib import lmap
from misc_lib import SuccessCounter


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


def get_eval_candidates(split):
    # split -> claims
    d_ids = load_claim_ids_for_split(split)
    claims: List[Dict] = get_claims_from_ids(d_ids)
    tokenizer = PCTokenizer()

    def get_candidates(c: Dict) -> Tuple[int, List[Dict]]:
        cid = c["cId"]
        assert type(cid) == int
        claim_text = c["text"]
        top_k = 50
        lucene_results = es_helper.get_perspective_from_pool(claim_text, top_k)
        candidate_list = []
        for rank, (_text, _pid, _score) in enumerate(lucene_results):
            rationale = "es_rank={} , es_score={}".format(rank, _score)
            p_entry = {
                'cid': cid,
                'pid': _pid,
                'claim_text': claim_text,
                'perspective_text': _text,
                'p_tokens': tokenizer.tokenize_stem(_text),
                'rationale': rationale,
            }
            candidate_list.append(p_entry)
        return cid, candidate_list

    candidates: List[Tuple[int, List[Dict]]] = lmap(get_candidates, claims)
    return candidates


def get_eval_candidates_from_pickle(split) -> List[Tuple[int, List[Dict]]]:
    return load_from_pickle("pc_candidates_{}".format(split))


def get_eval_candidate_as_pids(split) -> List[Tuple[int, List[int]]]:
    full_data: List[Tuple[int, List[Dict]]] = load_from_pickle("pc_candidates_{}".format(split))

    def convert(e) -> Tuple[int, List[int]]:
        cid, p_list = e
        return cid, lmap(lambda p: p['pid'], p_list)

    out_data = lmap(convert, full_data)
    return out_data


def precache():
    for split in splits:
        c = get_eval_candidates(split)
        save_to_pickle(c, "pc_candidates_{}".format(split))


def predict_from_dict(score_d: Dict[CPIDPair, float],
                      candidates: List[Tuple[int, List[Dict]]],
                      top_k,
                      ) -> List[Tuple[int, List[Dict]]]:
    suc_count = SuccessCounter()

    def rank(e: Tuple[int, List[Dict]]):
        cid, p_list = e
        scored_p_list: List[Dict] = []
        for p in p_list:
            pid = int(p['pid'])
            query_id = CPIDPair((cid, pid))

            if query_id in score_d:
                score = score_d[query_id]
                suc_count.suc()
            else:
                score = -2
                suc_count.fail()
            p['score'] = score
            scored_p_list.append(p)

        scored_p_list.sort(key=lambda x: x['score'], reverse=True)
        return cid, scored_p_list[:top_k]

    predictions = lmap(rank, candidates)
    print("{} found of {}".format(suc_count.get_suc(), suc_count.get_total()))
    return predictions


if __name__ == "__main__":
    save_dev_candidate()

