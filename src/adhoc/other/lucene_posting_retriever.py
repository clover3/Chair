from collections import defaultdict, Counter
from typing import List, Iterable, Callable, Dict, Tuple, Set

from adhoc.other.bm25t_retriever import IndexReaderIF
from adhoc.other.postings_search_dev import join_postings_naive
from adhoc.retriever_if import RetrieverIF
from misc_lib import get_second, dict_to_tuple_list
from trainer_v2.chair_logging import c_log


class LuceneBackBM25T_Retriever(RetrieverIF):
    def __init__(
            self,
            index_reader: IndexReaderIF,
            scoring_fn,
            tokenize_fn,
            translate_doc_id_fn,
            table: Dict[str, Dict[str, float]],
    ):
        self.index_reader = index_reader
        self.scoring_fn = scoring_fn
        self.tokenize_fn = tokenize_fn
        self.translate_doc_id_fn = translate_doc_id_fn
        self.join_postings_fn = join_postings_naive

        self.extension_term_set_d: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for q_term, entries in table.items():
            d_term_score_pairs: List[Tuple[str, float]] = dict_to_tuple_list(entries)
            d_term_score_pairs.sort(key=get_second, reverse=True)
            self.extension_term_set_d[q_term] = d_term_score_pairs

    def retrieve(self, query, n_retrieve=1000) -> List[Tuple[str, float]]:
        ret = self._retrieve_inner(query, n_retrieve)
        inn_doc_ids, scores = zip(*ret)

        ext_doc_ids = list(map(self.translate_doc_id_fn, inn_doc_ids))
        return list(zip(ext_doc_ids, scores))

    def _retrieve_inner(self, query, n_retrieve=1000) -> List[Tuple[int, float]]:
        c_log.debug("Query: %s", query)
        q_tokens = self.tokenize_fn(query)
        q_tf = Counter(q_tokens)
        q_terms = list(q_tf.keys())

        q_term_df_pairs = []
        for t in q_terms:
            df = self.index_reader.get_df(t)
            q_term_df_pairs.append((t, df))

        # Search rare query term first
        q_term_df_pairs.sort(key=get_second)

        # Compute how much score can be gained for remaining terms
        max_gain_per_term = []
        for q_term, qterm_df in q_term_df_pairs:
            assumed_tf = 10
            assumed_dl = 10
            qf = q_tf[q_term]
            max_gain_per_term.append(self.scoring_fn(assumed_tf, qf, assumed_dl, qterm_df))

        c_log.debug("max_gain_per_term: {}".format(str(max_gain_per_term)))

        doc_score: Dict[int, float] = Counter()
        for idx, (q_term, qterm_df) in enumerate(q_term_df_pairs):
            qf = q_tf[q_term]
            max_tf: Dict[str, float] = Counter()
            c_log.debug("Query term %s", q_term)

            c_log.debug("Request postings")
            target_term_postings = self.index_reader.get_postings(q_term)
            c_log.debug("Update counts")
            for doc_id, cnt in target_term_postings:
                max_tf[doc_id] = cnt

            target_term_posting_len = len(target_term_postings)

            total_posting_len = 0
            c_log.debug("Query term %s has %d extensions", q_term, len(self.extension_term_set_d[q_term]))
            for matching_term, match_score in self.extension_term_set_d[q_term]:
                c_log.debug("Request postings for %s", matching_term)
                postings = self.index_reader.get_postings(matching_term)
                c_log.debug("Update counts")
                total_posting_len += len(postings)
                for doc_id, cnt in postings:
                    if doc_id in max_tf:
                        # If terms are iterated by lower term match score, so max_tf[doc_id] > match_score
                        pass
                    else:
                        max_tf[doc_id] = cnt if match_score > 1.0 else match_score

            c_log.debug("Searched %d original posting, %d extended posting", target_term_posting_len, total_posting_len)
            qdf = target_term_posting_len
            min_score = 1000 * 1000
            for doc_id, cnt in max_tf.items():
                tf = cnt
                dl = self.index_reader.get_dl(doc_id)
                doc_score[doc_id] += self.scoring_fn(tf, qf, dl, qdf)
                cur_score = doc_score[doc_id]
                min_score = min(min_score, cur_score)
            c_log.debug("Done score updates")

            # Drop not-promising docs
            if len(doc_score) > n_retrieve:
                c_log.debug("Checking filtering")
                doc_score_pair_list = list(doc_score.items())
                doc_score_pair_list.sort(key=get_second, reverse=True)
                max_future_gain = sum(max_gain_per_term[idx+1:])
                _nth_doc, nth_doc_score = doc_score_pair_list[n_retrieve-1]
                drop_threshold: float = nth_doc_score - max_future_gain
                if drop_threshold > 0:
                    new_doc_score: dict[int, float] = Counter()
                    for doc_id, score in doc_score_pair_list:
                        if score >= drop_threshold:
                            new_doc_score[doc_id] = score
                        else:
                            break
                    c_log.debug("Keep %d docs from %d", len(new_doc_score), len(doc_score_pair_list))
                    doc_score = new_doc_score

        doc_score_pair_list: List[Tuple[int, float]] = list(doc_score.items())
        doc_score_pair_list.sort(key=get_second, reverse=True)
        return doc_score_pair_list

