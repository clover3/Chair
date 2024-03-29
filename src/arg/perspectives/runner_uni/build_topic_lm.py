from collections import Counter
from typing import List, NamedTuple, Iterable

from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import get_claim_perspective_id_dict, load_claims_for_sub_split, load_claim_ids_for_split, \
    get_claims_from_ids
from adhoc.kn_tokenizer import KrovetzNLTKTokenizer
from arg.perspectives.split_helper import train_split
from list_lib import lmap, foreach, left
from models.classic.lm_util import get_lm_log, subtract, least_common, smooth, average_counters, tokens_to_freq


class ClaimLM(NamedTuple):
    cid: int
    claim: str
    LM: Counter


def build_gold_claim_lm_train() -> List[ClaimLM]:
    # load claims and perspectives
    # Calculate term frequency for each terms.
    claims, val = train_split()
    return build_gold_lms(claims)


def build_gold_lms_for_sub_split(sub_split) -> List[ClaimLM]:
    claims = load_claims_for_sub_split(sub_split)

    return build_gold_lms(claims)


def build_gold_lms_for_split(split) -> List[ClaimLM]:
    d_ids: Iterable[int] = load_claim_ids_for_split(split)
    claims = get_claims_from_ids(d_ids)
    return build_gold_lms(claims)


def build_gold_lms(claims) -> List[ClaimLM]:
    gold = get_claim_perspective_id_dict()
    tokenizer = KrovetzNLTKTokenizer()

    def get_cluster_lm(cluster: List[int]) -> Counter:
        p_text_list: List[str] = lmap(perspective_getter, cluster)
        tokens_list: List[List[str]] = lmap(tokenizer.tokenize_stem, p_text_list)
        counter_list = lmap(tokens_to_freq, tokens_list)
        counter = average_counters(counter_list)
        return counter

    def get_claim_lm(claim) -> ClaimLM:
        cid = claim["cId"]
        counter_list: List[Counter] = lmap(get_cluster_lm, gold[cid])
        counter: Counter = average_counters(counter_list)
        return ClaimLM(cid, claim['text'], counter)

    claim_lms = lmap(get_claim_lm, claims)
    return claim_lms


def build_baseline_lms(claims):
    tokenizer = KrovetzNLTKTokenizer()

    def get_claim_lm(claim):
        cid = claim["cId"]
        counter = tokens_to_freq(tokenizer.tokenize_stem(claim['text']))
        return ClaimLM(cid, claim['text'], counter)

    claim_lms = lmap(get_claim_lm, claims)
    return claim_lms


def build_and_show():
    claim_lms = build_gold_claim_lm_train()
    alpha = 0.1
    bg_lm = average_counters(lmap(lambda x: x.LM, claim_lms))

    def show(claim_lm: ClaimLM):
        print('----')
        print(claim_lm.claim)
        log_topic_lm = get_lm_log(smooth(claim_lm.LM, bg_lm, alpha))
        log_bg_lm = get_lm_log(bg_lm)
        log_odd: Counter = subtract(log_topic_lm, log_bg_lm)

        for k, v in claim_lm.LM.most_common(50):
            print(k, v)

        s = "\t".join(left(claim_lm.LM.most_common(10)))
        print("LM freq: ", s)
        print(s)

        s = "\t".join(left(log_odd.most_common(30)))
        print("Log odd top", s)

        s = "\t".join(left(least_common(log_odd, 10)))
        print("Log odd bottom", s)

    foreach(show, claim_lms[:10])


if __name__ == "__main__":
    build_and_show()