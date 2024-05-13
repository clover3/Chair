from collections import Counter
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

from adhoc.other.ql.ql_formula import ql_scoring_from_p_qf
from trainer_v2.chair_logging import c_log

from adhoc.conf_helper import load_omega_config, QLIndexResource
from adhoc.other.bm25_retriever_helper import get_tokenize_fn
from adhoc.other.ql_retriever_helper import load_ql_stats, build_ql_scoring_fn_from_conf
from misc_lib import TimeEstimatorOpt
from trainer_v2.per_project.transparency.mmp.table_readers import load_align_scores


# It has different constructor than BM25T_3
class QLRerank:
    def __init__(
            self,
            scoring_fn,
            tokenize_fn,
            bg_prob_d: Dict[str, float]
    ):
        self.tokenize_fn = tokenize_fn
        self.bg_prob_d = bg_prob_d
        self.scoring_fn = scoring_fn
        self.q_term_not_found = set()

    def score(self, query, text):
        q_terms = self.tokenize_fn(query)
        t_terms = self.tokenize_fn(text)
        q_tf = Counter(q_terms)
        t_tf = Counter(t_terms)
        score = self.score_from_tfs(q_tf, t_tf)
        return score

    def score_batch(self, qd_list, show_time=False) -> List[float]:
        output: List[float] = []
        ticker = TimeEstimatorOpt(len(qd_list) if show_time else None)
        for q, d in qd_list:
            s = self.score(q, d)
            output.append(s)
            ticker.tick()
        return output

    def _get_q_term_bg_prob(self, q_term):
        try:
            return self.bg_prob_d[q_term]
        except KeyError:
            if q_term not in self.q_term_not_found:
                self.q_term_not_found.add(q_term)
                c_log.warn("Term %s is not found", q_term)
            return 1 / (100 * 1000 * 1000)

    def score_from_tfs(self, q_tf: Dict[str, int], t_tf: Dict[str, int]):
        dl = sum(t_tf.values())
        score_sum = 0
        for q_term, q_cnt in q_tf.items():
            exact_cnt = t_tf[q_term]
            qt_bg_prob = self._get_q_term_bg_prob(q_term)
            score_sum += self.scoring_fn(exact_cnt, q_cnt, dl, qt_bg_prob)
        return score_sum


class QLTRerank(QLRerank):
    def __init__(
            self,
            mapping: Dict[str, Dict[str, float]],
            scoring_fn,
            tokenize_fn,
            bg_prob_d: Dict[str, float],
            mu: float,
    ):
        super(QLTRerank, self).__init__(scoring_fn, tokenize_fn, bg_prob_d)
        self.mu = mu
        self.relevant_terms = mapping

    def score_from_tfs(self, q_tf: Dict[str, int], t_tf: Dict[str, int]):
        dl = sum(t_tf.values())
        score_sum = 0
        for q_term, q_cnt in q_tf.items():
            exact_cnt = t_tf[q_term]

            p_qt_w_sum = 0
            self_tp = 1
            p_qt_w = self_tp * exact_cnt / dl
            p_qt_w_sum += p_qt_w

            orig_p = p_qt_w_sum

            translation_term_set: Dict[str, float] = self.relevant_terms[q_term]
            expansion_tf: List[tuple[int, float]] = []
            for t, cnt in t_tf.items():
                if t in translation_term_set:
                    expansion_tf.append((cnt, translation_term_set[t]))

            for tf, tp in expansion_tf:
                p_qt_w = tp * tf / dl
                p_qt_w_sum += p_qt_w

            after_p = p_qt_w_sum
            if after_p != orig_p:
                c_log.debug("{} -> {}".format(orig_p, after_p))
            qt_bg_prob = self._get_q_term_bg_prob(q_term)
            score_sum += ql_scoring_from_p_qf(p_qt_w_sum, dl, qt_bg_prob, self.mu)
        return score_sum


def get_ql_scorer_fn(conf):
    ql_conf = load_omega_config(conf.ql_conf_path, set_project_root=True)
    tokenize_fn = get_tokenize_fn(ql_conf)
    _avdl, _cdf, bg_prob, dl = load_ql_stats(ql_conf)
    scoring_fn = build_ql_scoring_fn_from_conf(ql_conf)

    ql_rerank = QLRerank(scoring_fn, tokenize_fn, bg_prob)
    return ql_rerank.score_batch


def get_ql_t_scorer_fn(conf):
    value_mapping: Dict[str, Dict[str, float]] = load_align_scores(conf.table_path)
    ql_conf = load_omega_config(conf.ql_conf_path, set_project_root=True)
    tokenize_fn = get_tokenize_fn(ql_conf)
    _avdl, _cdf, bg_prob, dl = load_ql_stats(ql_conf)
    dummy_scoring_fn = build_ql_scoring_fn_from_conf(ql_conf)
    ql_rerank = QLTRerank(value_mapping, dummy_scoring_fn, tokenize_fn, bg_prob, ql_conf.mu)
    return ql_rerank.score_batch
