from collections import Counter
from typing import List, Dict

from arg.perspectives.pc_tokenizer import PCTokenizer
from arg.qck.decl import QKUnit, KDP
from list_lib import lmap, lfilter, right
from misc_lib import average
from models.classic.lm_util import average_counters, get_lm_log, smooth, subtract
from models.classic.stopword import load_stopwords_for_query


def filter_qk(qk_candidate: List[QKUnit], query_lms: Dict[str, Counter]) -> List[QKUnit]:
    bg_lm = average_counters(list(query_lms.values()))
    log_bg_lm = get_lm_log(bg_lm)
    alpha = 0.5
    stopwords = load_stopwords_for_query()
    tokenizer = PCTokenizer()

    filtered_qk_list: List[QKUnit] = []
    for query, k_candidates in qk_candidate:
        query_lm: Counter = query_lms[query.query_id]
        log_topic_lm = get_lm_log(smooth(query_lm, bg_lm, alpha))
        log_odd: Counter = subtract(log_topic_lm, log_bg_lm)

        def get_kdp_score(kdp: KDP) -> float:
            def get_score(t):
                if t in stopwords:
                    return 0
                try:
                    return log_odd[tokenizer.stemmer.stem(t)]
                except UnicodeDecodeError:
                    return 0

            return average(lmap(get_score, kdp.tokens))

        good_kdps: List[KDP] = lfilter(lambda kdp: get_kdp_score(kdp) > 0, k_candidates)
        filtered_qk_list.append((query, good_kdps))

    n_no_kdp_query = sum(lmap(lambda l: 1 if not l else 0, right(filtered_qk_list)))
    print("{} queries, {} has no kdp ".format(len(qk_candidate), n_no_kdp_query))
    return filtered_qk_list
