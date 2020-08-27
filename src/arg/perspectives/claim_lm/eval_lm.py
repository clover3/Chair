from typing import List, Callable, Dict, Tuple

from arg.perspectives.claim_lm.debug_lm import debug_failture
from arg.perspectives.claim_lm.lm_predict import get_lm_scorer
from arg.perspectives.claim_lm.passage_to_lm import get_train_passage_a_lms
from arg.perspectives.collection_based_classifier import NamedNumber
from arg.perspectives.eval_helper import get_eval_candidates, get_eval_candidates_from_pickle
from arg.perspectives.evaluate import evaluate_map
from arg.perspectives.runner_uni.build_topic_lm import ClaimLM
from list_lib import lmap


def eval_raw(claim_lm_list: List[ClaimLM], split):
    # load pre-computed perspectives
    candidates = get_eval_candidates(split)
    return eval_claim_lm_list(candidates, claim_lm_list)


def eval_fast(claim_lm_list: List[ClaimLM], split):
    # load pre-computed perspectives
    candidates: List[Tuple[int, List[Dict]]] = get_eval_candidates_from_pickle(split)
    return eval_claim_lm_list(candidates, claim_lm_list)


def eval_claim_lm_list(candidates: List[Tuple[int, List[Dict]]],
                       claim_lm_list):
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


def eval_debug(claim_lm_list: List[ClaimLM], split):
    # load pre-computed perspectives
    candidates = get_eval_candidates_from_pickle(split)
    return debug_eval_claim_lm_list(candidates, claim_lm_list)


def debug_eval_claim_lm_list(candidates, claim_lm_list):
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
    return debug_failture(predictions)


def main():
    claim_lms = get_train_passage_a_lms()
    r = eval_debug(claim_lms, "train")
    print(r)


if __name__ == "__main__":
    main()
