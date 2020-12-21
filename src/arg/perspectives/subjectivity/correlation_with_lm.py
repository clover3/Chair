import sys
from collections import Counter
from typing import List, Dict, Tuple

from scipy.stats import pearsonr

from arg.perspectives.clueweb_db import load_doc
from arg.perspectives.pc_tokenizer import PCTokenizer
from arg.perspectives.runner_uni.build_topic_lm import build_gold_lms_for_sub_split, ClaimLM
from datastore.interface import preload_man
from datastore.table_names import TokenizedCluewebDoc
from galagos.parse import load_galago_ranked_list
from galagos.types import SimpleRankedListEntry
from list_lib import lmap
from models.classic.lm_util import average_counters, get_lm_log, smooth, subtract
from models.classic.stopword import load_stopwords_for_query


def load_subjectivity(path):
    d = {}
    for line in open(path, "r"):
        doc_id, num_subj, num_sent = line.split("\t")
        num_subj = int(num_subj)
        num_sent = int(num_sent)
        d[doc_id] = num_subj, num_sent
    return d


def main():
    split = "train"
    subjectivity_path = sys.argv[1]
    q_res_path = sys.argv[2]
    ranked_list: Dict[str, List[SimpleRankedListEntry]] = load_galago_ranked_list(q_res_path)

    # load LM
    claim_lms: List[ClaimLM] = build_gold_lms_for_sub_split(split)
    bg_lm = average_counters(lmap(lambda x: x.LM, claim_lms))
    log_bg_lm = get_lm_log(bg_lm)
    alpha = 0.1
    stopwords = load_stopwords_for_query()
    # load subjectivity predictions.
    subj_d: Dict[str, Tuple[int, int]] = load_subjectivity(subjectivity_path)
    doc_ids = subj_d.keys()
    preload_man.preload(TokenizedCluewebDoc, doc_ids)
    tokenizer = PCTokenizer()

    lm_scores = []
    rates = []
    num_subj_list = []
    num_sent_list = []
    for claim_lm in claim_lms:
        qid = str(claim_lm.cid)
        log_topic_lm = get_lm_log(smooth(claim_lm.LM, bg_lm, alpha))
        log_odd: Counter = subtract(log_topic_lm, log_bg_lm)

        def get_passage_score(p):
            def get_score(t):
                if t in stopwords:
                    return 0
                return log_odd[tokenizer.stemmer.stem(t)]

            return sum([get_score(t) for t in p]) / len(p) if len(p) > 0 else 0

        for entry in ranked_list[qid]:
            if entry.doc_id in subj_d:
                tokens = load_doc(entry.doc_id)
                assert type(tokens[0]) == str
                lm_score = get_passage_score(tokens)
                num_subj, num_sent = subj_d[entry.doc_id]
                rate = num_subj / num_sent
                lm_scores.append(lm_score)
                rates.append(rate)
                num_subj_list.append(num_subj)
                num_sent_list.append(num_sent)



    print("lm scores correlation with ")
    print("rates: ", pearsonr(lm_scores, rates))
    print("num subj: ", pearsonr(lm_scores, num_subj_list))
    print("num sent: ", pearsonr(lm_scores, num_sent_list))


if __name__ == "__main__":
    main()