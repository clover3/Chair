from collections import Counter

import tensorflow as tf

from adhoc.other.index_reader_wrap import IndexReaderIF, QLIndexReaderIF
from adhoc.retriever_if import RetrieverIF
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator, Union

from misc_lib import get_second
from trainer_v2.chair_logging import c_log
DocID = Union[int, str]


class QLRetriever(RetrieverIF):
    def __init__(
            self,
            index_reader: QLIndexReaderIF,
            scoring_fn,
            tokenize_fn,
            stopwords,
    ):
        self.index_reader = index_reader
        self.scoring_fn = scoring_fn
        self.tokenize_fn = tokenize_fn
        self.stopwords = set(stopwords)
        self.q_term_not_found = set()

    def retrieve(self, query, n_retrieve=1000) -> List[Tuple[str, float]]:
        ret = self._retrieve_inner(query, n_retrieve)
        output: List[Tuple[str, float]] = []
        for doc_id, score in ret:
            if type(doc_id) != str:
                doc_id = str(doc_id)
            output.append((doc_id, score))
        return output

    def _get_q_term_bg_prob(self, q_term):
        try:
            return self.index_reader.get_bg_prob(q_term)
        except KeyError:
            if q_term not in self.q_term_not_found:
                self.q_term_not_found.add(q_term)
                c_log.warn("Term %s is not found", q_term)
            return 1 / (100 * 1000 * 1000)

    def _retrieve_inner(self, query, n_retrieve=1000) -> List[Tuple[DocID, float]]:
        c_log.debug("Query: %s", query)
        q_tokens = self.tokenize_fn(query)
        q_tf = Counter(q_tokens)
        q_terms = list(q_tf.keys())

        q_terms = [term for term in q_terms if term not in self.stopwords]
        q_term_bg_prob_pairs = []
        for t in q_terms:
            qt_bg_prob = self._get_q_term_bg_prob(t)
            q_term_bg_prob_pairs.append((t, qt_bg_prob))

        # Search rare query term first
        q_term_bg_prob_pairs.sort(key=get_second)

        # Compute how much score can be gained for remaining terms
        max_gain_per_term = []
        for q_term, q_term_bg_prob in q_term_bg_prob_pairs:
            assumed_tf = 10
            assumed_dl = 10
            qf = q_tf[q_term]
            max_gain_per_term.append(self.scoring_fn(assumed_tf, qf, assumed_dl, q_term_bg_prob))

        c_log.debug("max_gain_per_term: {}".format(str(max_gain_per_term)))

        doc_score: Dict[DocID, float] = Counter()
        for idx, (q_term, q_term_bg_prob) in enumerate(q_term_bg_prob_pairs):
            qf = q_tf[q_term]
            max_tf: Dict[str, float] = Counter()
            c_log.debug("Query term %s", q_term)

            def get_posting_local(q_term):
                c_log.debug("Request postings")
                postings = self.index_reader.get_postings(q_term)
                return postings

            target_term_postings = get_posting_local(q_term)
            c_log.debug("Update counts")
            for doc_id, cnt in target_term_postings:
                max_tf[doc_id] = cnt

            min_score = 1000 * 1000
            for doc_id, cnt in max_tf.items():
                tf = cnt
                dl = self.index_reader.get_dl(doc_id)
                doc_score[doc_id] += self.scoring_fn(tf, qf, dl, q_term_bg_prob)
                cur_score = doc_score[doc_id]
                min_score = min(min_score, cur_score)
            c_log.debug("Done score updates")

        doc_score_pair_list: List[Tuple[DocID, float]] = list(doc_score.items())
        doc_score_pair_list.sort(key=get_second, reverse=True)
        return doc_score_pair_list

