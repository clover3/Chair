from collections import Counter
from typing import List, Dict, Tuple

from arg.perspectives.claim_lm.passage_common import iterate_passages
from arg.perspectives.claim_lm.show_docs_per_claim import preload_docs
from arg.perspectives.load import load_train_claim_ids, get_claims_from_ids, load_dev_claim_ids
from arg.perspectives.pc_tokenizer import PCTokenizer
from arg.perspectives.runner_uni.build_topic_lm import build_gold_lms
from base_type import FilePath
from cache import save_to_pickle
from galagos.parse import load_galago_ranked_list
from galagos.types import SimpleRankedListEntry
from list_lib import lmap, lfilter
from models.classic.lm_util import get_lm_log, subtract, smooth, average_counters
from models.classic.stopword import load_stopwords_for_query


def a_relevant_candidate(save_name, q_res_path, claims):
    top_n = 10
    ranked_list: Dict[str, List[SimpleRankedListEntry]] = load_galago_ranked_list(q_res_path)
    preload_docs(ranked_list, claims, top_n)
    all_passages = []
    entries = []

    all_docs = 0
    for c in claims:
        q_res: List[SimpleRankedListEntry] = ranked_list[str(c['cId'])]
        claim_text = c['text']
        def get_passage_score(dummy):
            return 0
        passages: List[Tuple[List[str], float]] = iterate_passages(q_res, top_n, get_passage_score)
        all_docs += len(passages)
        all_passages.extend(passages)
        entries.append((c, passages))

    print("{} claims. {} docs ".format(len(claims), all_docs))
    data = entries, all_passages
    save_to_pickle(data, save_name)


def a_relevant(save_name, q_res_path, claims):
    top_n = 10

    ranked_list: Dict[str, List[SimpleRankedListEntry]] = load_galago_ranked_list(q_res_path)
    preload_docs(ranked_list, claims, top_n)
    claim_lms = build_gold_lms(claims)
    claim_lms_d = {lm.cid: lm for lm in claim_lms}
    bg_lm = average_counters(lmap(lambda x: x.LM, claim_lms))
    log_bg_lm = get_lm_log(bg_lm)

    stopwords = load_stopwords_for_query()
    alpha = 0.5

    tokenizer = PCTokenizer()
    all_passages = []
    entries = []
    num_pos_sum = 0
    num_pos_exists = 0

    for c in claims:
        q_res: List[SimpleRankedListEntry] = ranked_list[str(c['cId'])]
        claim_lm = claim_lms_d[c['cId']]
        log_topic_lm = get_lm_log(smooth(claim_lm.LM, bg_lm, alpha))
        log_odd: Counter = subtract(log_topic_lm, log_bg_lm)

        claim_text = c['text']
        claim_tokens = tokenizer.tokenize_stem(claim_text)

        scores = []
        for t in claim_tokens:
            if t in log_odd:
                scores.append(log_odd[t])

        def get_passage_score(p):
            def get_score(t):
                if t in stopwords:
                    return 0
                return log_odd[tokenizer.stemmer.stem(t)]

            return sum([get_score(t) for t in p]) / len(p) if len(p) > 0 else 0

        passages = iterate_passages(q_res, top_n, get_passage_score)
        num_pos = len(lfilter(lambda x: x[1] > 0, passages))
        num_pos_sum += num_pos
        if num_pos > 0:
            num_pos_exists += 1

        all_passages.extend(passages)
        entries.append((c, passages))

    print("{} claims. {} docs on {} claims".format(len(claims), num_pos_sum, num_pos_exists))

    data = entries, all_passages

    save_to_pickle(data, save_name)


def save_train():
    save_name = "pc_train_a_passages"
    q_res_path = FilePath("/mnt/nfs/work3/youngwookim/data/perspective/train_claim/q_res_100")
    d_ids = list(load_train_claim_ids())
    claims: List[Dict] = get_claims_from_ids(d_ids)
    a_relevant(save_name, q_res_path, claims)


def save_dev():
    save_name = "pc_dev_a_passages"
    q_res_path = FilePath("/mnt/nfs/work3/youngwookim/data/perspective/dev_claim/q_res_100")
    d_ids = list(load_dev_claim_ids())
    claims: List[Dict] = get_claims_from_ids(d_ids)
    a_relevant_candidate(save_name, q_res_path, claims)


if __name__ == "__main__":
    save_dev()
