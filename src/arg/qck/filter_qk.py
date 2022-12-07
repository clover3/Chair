from collections import Counter
from typing import List, Dict

import math

from arg.perspectives.kn_tokenizer import KrovetzNLTKTokenizer
from arg.qck.decl import QKUnit, KDP
from list_lib import lmap, lfilter, right
from misc_lib import average, TimeEstimator
from models.classic.lm_util import average_counters, get_lm_log, tokens_to_freq
from models.classic.stopword import load_stopwords_for_query


def text_list_to_lm(tokenizer: KrovetzNLTKTokenizer, text_list: List[str]) -> Counter:
    tokens_list: List[List[str]] = lmap(tokenizer.tokenize_stem, text_list)
    counter_list = lmap(tokens_to_freq, tokens_list)
    counter = average_counters(counter_list)
    return counter


class LMScorer:
    def __init__(self, query_lms: Dict[str, Counter], alpha=0.5):
        self.query_lms = query_lms
        bg_lm = average_counters(list(query_lms.values()))
        self.bg_lm = bg_lm
        self.log_bg_lm: Counter = get_lm_log(bg_lm)
        self.alpha = alpha
        self.log_odd_d: Dict[str, Counter] = {k: Counter() for k in query_lms.keys()}
        self.stopwords = load_stopwords_for_query()
        self.tokenizer = KrovetzNLTKTokenizer()

    def score(self, query_id, raw_tokens) -> float:
        stemmed_tokens = self.filter_and_stem(raw_tokens)
        return self._get_score_from_stemmed_tokens(query_id, stemmed_tokens)

    def filter_and_stem(self, tokens):
        stemmed_tokens = []
        for t in tokens:
            if t in self.stopwords:
                pass
            else:
                try:
                    stemmed_t = self.tokenizer.stemmer.stem(t)
                    stemmed_tokens.append(stemmed_t)
                except UnicodeDecodeError:
                    pass
        return stemmed_tokens

    def score_text(self, query_id, text):
        tokens = self.tokenizer.tokenize_stem(text)
        tokens = list([t for t in tokens if t not in self.stopwords])
        return self._get_score_from_stemmed_tokens(query_id, tokens)

    def _get_score_from_stemmed_tokens(self, query_id, tokens) -> float:
        log_odd_d: Counter = self.log_odd_d[query_id]
        lm = self.query_lms[query_id]

        def get_score(token: str) -> float:
            if token in log_odd_d:
                return log_odd_d[token]

            if token in lm or token in self.bg_lm:
                prob_pos = lm[token] * (1 - self.alpha) + self.bg_lm[token] * self.alpha
                pos_log = math.log(prob_pos)
            else:
                pos_log = 0
            score = pos_log - self.log_bg_lm[token]
            log_odd_d[token] = score
            return score

        return average(lmap(get_score, tokens))


def filter_qk(qk_candidate: List[QKUnit], query_lms: Dict[str, Counter], alpha=0.5) -> List[QKUnit]:
    scorer = LMScorer(query_lms, alpha)

    filtered_qk_list: List[QKUnit] = []
    ticker = TimeEstimator(len(qk_candidate))
    for query, k_candidates in qk_candidate:
        def get_kdp_score(kdp: KDP) -> float:
            return scorer.score(query.query_id, kdp.tokens)

        good_kdps: List[KDP] = lfilter(lambda kdp: get_kdp_score(kdp) > 0, k_candidates)
        filtered_qk_list.append((query, good_kdps))
        ticker.tick()

    n_no_kdp_query = sum(lmap(lambda l: 1 if not l else 0, right(filtered_qk_list)))
    print("{} queries, {} has no kdp ".format(len(qk_candidate), n_no_kdp_query))
    return filtered_qk_list


def filter_qk_rel(qk_candidate: List[QKUnit],
                  query_lms: Dict[str, Counter],
                  top_n=50) -> List[QKUnit]:
    scorer = LMScorer(query_lms)

    filtered_qk_list: List[QKUnit] = []
    ticker = TimeEstimator(len(qk_candidate))
    for query, k_candidates in qk_candidate:
        def get_kdp_score(kdp: KDP) -> float:
            return scorer.score(query.query_id, kdp.tokens)

        k_candidates.sort(key=get_kdp_score, reverse=True)
        good_kdps: List[KDP] = k_candidates[:top_n]
        filtered_qk_list.append((query, good_kdps))
        ticker.tick()

    n_no_kdp_query = sum(lmap(lambda l: 1 if not l else 0, right(filtered_qk_list)))
    print("{} queries, {} has no kdp ".format(len(qk_candidate), n_no_kdp_query))
    return filtered_qk_list
