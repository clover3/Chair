from typing import List, Dict, Counter

from arg.perspectives.claim_lm.show_docs_per_claim import preload_docs
from arg.perspectives.clueweb_db import load_doc
from arg.perspectives.load import load_train_claim_ids, get_claims_from_ids
from arg.perspectives.kn_tokenizer import KrovetzNLTKTokenizer
from arg.perspectives.runner_uni.build_topic_lm import build_gold_lms
from base_type import FilePath
from galagos.parse import load_galago_ranked_list
from galagos.types import SimpleRankedListEntry
from list_lib import lmap
from misc_lib import average
from models.classic.lm_util import get_lm_log, smooth, subtract, average_counters
from models.classic.stopword import load_stopwords_for_query
from tab_print import print_table


def get_doc_score(doc, get_passage_score):
    idx = 0
    window_size = 300
    scores = []
    while idx < len(doc):
        p = doc[idx:idx + window_size]
        score = get_passage_score(p)
        scores.append(score)
        idx += window_size
    return scores


def a_relevant():
    d_ids = list(load_train_claim_ids())
    claims: List[Dict] = get_claims_from_ids(d_ids)
    claim_lms = build_gold_lms(claims)
    claim_lms_d = {lm.cid: lm for lm in claim_lms}
    bg_lm = average_counters(lmap(lambda x: x.LM, claim_lms))
    log_bg_lm = get_lm_log(bg_lm)

    claims = claims[:10]
    top_n = 100
    q_res_path = FilePath("/mnt/nfs/work3/youngwookim/data/perspective/train_claim/q_res_100")
    ranked_list: Dict[str, List[SimpleRankedListEntry]] = load_galago_ranked_list(q_res_path)
    preload_docs(ranked_list, claims, top_n)

    stopwords = load_stopwords_for_query()
    alpha = 0.7

    tokenizer = KrovetzNLTKTokenizer()
    for c in claims:
        q_res: List[SimpleRankedListEntry] = ranked_list[str(c['cId'])]
        claim_lm = claim_lms_d[c['cId']]
        log_topic_lm = get_lm_log(smooth(claim_lm.LM, bg_lm, alpha))
        log_odd: Counter = subtract(log_topic_lm, log_bg_lm)

        def get_passage_score(p):
            def get_score(t):
                if t in stopwords:
                    return 0
                return log_odd[tokenizer.stemmer.stem(t)]

            return sum([get_score(t) for t in p]) / len(p) if len(p) > 0 else 0

        docs = []
        for i in range(top_n):
            try:
                doc = load_doc(q_res[i].doc_id)
                docs.append(doc)
            except KeyError:
                docs.append(None)
                pass

        print(c['text'])
        rows = []
        for rank, doc in enumerate(docs):
            if doc is None:
                rows.append((rank, "-", "-"))
                continue

            scores = get_doc_score(doc, get_passage_score)
            avg_score = average(scores)
            max_score = max(scores)
            rows.append((rank, avg_score, max_score))

        print_table(rows)


if __name__ == "__main__":
    a_relevant()
