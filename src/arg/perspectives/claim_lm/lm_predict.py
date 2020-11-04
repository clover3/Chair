from collections import Counter
from typing import List, Dict, Tuple

from arg.perspectives.collection_based_classifier import NamedNumber, predict_interface
from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.pc_tokenizer import PCTokenizer
from arg.perspectives.runner_uni.build_topic_lm import ClaimLM
from list_lib import lmap
from models.classic.lm_util import get_log_odd, merge_lms


def get_lm_scorer(claim_lms: List[ClaimLM], alpha):
    bg_lm = merge_lms(lmap(lambda x: x.LM, claim_lms))
    claim_log_odds_dict: Dict[int, Counter] = {c_lm.cid: get_log_odd(c_lm, bg_lm, alpha) for c_lm in claim_lms}

    def scorer(claim_id: int, p_tokens: List[str]) -> NamedNumber:
        c_lm = claim_log_odds_dict[claim_id]
        reason = " ".join(["{0} ({1:.2f})".format(t, c_lm[t]) for t in p_tokens])
        score = sum([c_lm[t] for t in p_tokens])
        return NamedNumber(score, reason)
    return scorer



def predict_by_lm(claim_lms: List[ClaimLM],
                  claims,
                  top_k) -> List[Tuple[str, List[Dict]]]:

    alpha = 0.1
    bg_lm = merge_lms(lmap(lambda x: x.LM, claim_lms))
    tokenizer = PCTokenizer()
    print("Eval log odds")
    claim_log_odds_dict = {str(c_lm.cid): get_log_odd(c_lm, bg_lm, alpha) for c_lm in claim_lms}

    def scorer(lucene_score, query_id) -> NamedNumber:
        claim_id, p_id = query_id.split("_")
        p_text = perspective_getter(int(p_id))
        tokens = tokenizer.tokenize_stem(p_text)
        c_lm = claim_log_odds_dict[claim_id]
        reason = " ".join(["{0} ({1:.2f})".format(t, c_lm[t]) for t in tokens])
        score = sum([c_lm[t] for t in tokens])
        return NamedNumber(score, reason)

    r = predict_interface(claims, top_k, scorer)
    return r
