from typing import List, Callable, Dict, Tuple

from arg.perspectives import es_helper
from arg.perspectives.claim_lm.lm_predict import get_lm_scorer
from arg.perspectives.collection_based_classifier import NamedNumber
from arg.perspectives.evaluate import evaluate_map
from arg.perspectives.load import load_claim_ids_for_split, get_claims_from_ids, splits
from arg.perspectives.pc_tokenizer import PCTokenizer
from arg.perspectives.runner_uni.build_topic_lm import ClaimLM, build_claim_lm_trian
from cache import save_to_pickle, load_from_pickle
from list_lib import lmap


def eval_raw(claim_lm_list: List[ClaimLM], split):
    # load pre-computed perspectives
    candidates = get_eval_candidates(split)
    return eval_claim_lm_list(candidates, claim_lm_list)


def eval_fast(claim_lm_list: List[ClaimLM], split):
    # load pre-computed perspectives
    candidates = get_eval_candidates_from_pickle(split)
    return eval_claim_lm_list(candidates, claim_lm_list)


def eval_claim_lm_list(candidates, claim_lm_list):
    def rank(e: Tuple[int, List[Dict]]):
        cid, p_list = e
        scored_p_list: List[Dict] = []
        for p in p_list:
            p['score'] = scorer(cid, p['p_tokens'])
            scored_p_list.append(p)

        scored_p_list.sort(key=lambda x: x['score'], reverse=True)
        return cid, scored_p_list

    scorer: Callable[[int, List[str]], NamedNumber] = get_lm_scorer(claim_lm_list, 0.1)
    predictions = lmap(rank, candidates)
    return evaluate_map(predictions, False)


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


def get_eval_candidates_from_pickle(split):
    return load_from_pickle("pc_candidates_{}".format(split))


def precache():
    for split in splits:
        c = get_eval_candidates(split)
        save_to_pickle(c, "pc_candidates_{}".format(split))


def main():
    claim_lms = build_claim_lm_trian()
    r = eval_fast(claim_lms, "train")
    print(r)


if __name__ == "__main__":
    main()