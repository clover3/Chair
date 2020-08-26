from collections import Counter
from collections import Counter
from typing import List, Dict

from arg.perspectives.claim_lm.passage_common import score_passages
from arg.perspectives.claim_lm.show_docs_per_claim import preload_docs
from arg.perspectives.load import load_train_claim_ids, get_claims_from_ids
from arg.perspectives.pc_tokenizer import PCTokenizer
from arg.perspectives.runner_uni.build_topic_lm import build_gold_lms, merge_lms, get_lm_log, subtract, smooth
from base_type import FilePath
from cache import save_to_pickle
from galagos.parse import load_galago_ranked_list
from galagos.types import GalagoDocRankEntry
from list_lib import lmap, lfilter
from models.classic.stopword import load_stopwords_ex


def a_relevant():

    d_ids = list(load_train_claim_ids())
    claims: List[Dict] = get_claims_from_ids(d_ids)
    claims = claims
    top_n = 10
    q_res_path = FilePath("/mnt/nfs/work3/youngwookim/data/perspective/train_claim/q_res_100")
    ranked_list: Dict[str, List[GalagoDocRankEntry]] = load_galago_ranked_list(q_res_path)
    preload_docs(ranked_list, claims, top_n)
    claim_lms = build_gold_lms(claims)
    claim_lms_d = {lm.cid: lm for lm in claim_lms}
    bg_lm = merge_lms(lmap(lambda x: x.LM, claim_lms))
    log_bg_lm = get_lm_log(bg_lm)

    stopwords = load_stopwords_ex()
    alpha = 0.5

    tokenizer = PCTokenizer()
    all_passages = []
    entries = []
    num_pos_sum = 0
    num_pos_exists = 0

    for c in claims:
        q_res: List[GalagoDocRankEntry] = ranked_list[str(c['cId'])]
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

        passages = score_passages(q_res, top_n, get_passage_score)
        num_pos = len(lfilter(lambda x: x[1] > 0, passages))
        num_pos_sum += num_pos
        if num_pos > 0:
            num_pos_exists += 1

        all_passages.extend(passages)
        entries.append((c, passages))

    print("{} claims. {} docs on {} claims".format(len(claims), num_pos_sum, num_pos_exists))

    data = entries, all_passages
    save_to_pickle(data, "pc_train_a_passages")


if __name__ == "__main__":
    a_relevant()
