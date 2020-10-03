from collections import Counter
from typing import List, Dict, Tuple

from arg.perspectives import es_helper
from arg.perspectives.bm25_predict import get_bm25_module
from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import get_claims_from_ids, load_dev_claim_ids, load_claim_ids_for_split, splits, \
    get_claim_perspective_id_dict2
from arg.perspectives.pc_tokenizer import PCTokenizer
from cache import save_to_pickle
from list_lib import lmap, left


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


def precache():
    for split in splits:
        c = get_eval_candidates(split)
        save_to_pickle(c, "pc_candidates_{}".format(split))


if __name__ == "__main__":
    save_dev_candidate()


def get_extended_eval_candidate(split) -> Dict[int, List[int]]:
    bm25 = get_bm25_module()
    d_ids = load_claim_ids_for_split(split)
    claims: List[Dict] = get_claims_from_ids(d_ids)
    cid_to_pids: Dict[int, List[int]] = get_claim_perspective_id_dict2()
    tokenizer = PCTokenizer()

    def get_tf_idf(c: Counter):
        r = Counter()
        for t, cnt in c.items():
            tfidf = bm25.term_idf_factor(t) * cnt
            r[t] = tfidf
        return r

    def get_candidates(c: Dict) -> Tuple[int, List[int]]:
        cid = c["cId"]
        assert type(cid) == int
        claim_text = c["text"]
        claim_tokens = tokenizer.tokenize_stem(claim_text)
        top_k = 50
        lucene_results = es_helper.get_perspective_from_pool(claim_text, top_k)
        candidate_list: List[int] = []

        for rank, (_text, _pid, _score) in enumerate(lucene_results):
            candidate_list.append(_pid)

        gold_pids = cid_to_pids[int(cid)]
        hard_candidate = []
        mismatch_voca = Counter()
        for pid in gold_pids:
            if pid not in candidate_list:
                hard_candidate.append(pid)
                p_text = perspective_getter(pid)
                p_tokens = tokenizer.tokenize_stem(p_text)

                for t in p_tokens:
                    if t not in claim_tokens:
                        mismatch_voca[t] += 1

        candidate_list.extend(hard_candidate)
        mismatch_tf_idf = get_tf_idf(mismatch_voca)
        new_qterms = left(mismatch_tf_idf.most_common(30))
        lucene_results = es_helper.get_perspective_from_pool(" ".join(new_qterms), top_k)

        for rank, (_text, _pid, _score) in enumerate(lucene_results):
            if _pid not in candidate_list:
                candidate_list.append(_pid)

        return cid, candidate_list

    candidates: List[Tuple[int, List[int]]] = lmap(get_candidates, claims)
    return dict(candidates)

