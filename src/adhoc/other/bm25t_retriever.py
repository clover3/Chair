from collections import Counter, defaultdict
from typing import List, Iterable, Callable, Dict, Tuple, Set, TypeVar, Union

from adhoc.bm25_retriever import RetrieverIF
from adhoc.kn_tokenizer import KrovetzNLTKTokenizer
from list_lib import left
from misc_lib import get_second, dict_to_tuple_list
from typing import List, Iterable, Dict, Tuple, Set

from trainer_v2.chair_logging import c_log


class BM25T_Retriever(RetrieverIF):
    def __init__(
            self, inv_index, df, dl_d,
            scoring_fn,
            table: Dict[str, List[str]],
            mapping_val=0.1
    ):
        self.inv_index = inv_index
        self.scoring_fn = scoring_fn
        self.df = df
        self.tokenizer = KrovetzNLTKTokenizer(False)
        self.tokenize_fn = self.tokenizer.tokenize_stem
        self.dl_d = dl_d
        self.table: Dict[str, List[str]] = table
        self.mapping_val = mapping_val

    def get_low_df_terms(self, q_terms: Iterable[str], n_limit=10) -> List[str]:
        candidates = []
        for t in q_terms:
            df = self.df[t]
            candidates.append((t, df))

        candidates.sort(key=get_second)
        return left(candidates)[:n_limit]

    def get_posting(self, term):
        if term in self.inv_index:
            return self.inv_index[term]
        else:
            return []

    def get_extension_terms(self, term) -> List[str]:
        if term in self.table:
            return self.table[term]
        else:
            return []

    def retrieve(self, query) -> List[Tuple[str, float]]:
        q_tokens = self.tokenize_fn(query)
        q_tf = Counter(q_tokens)
        doc_score = Counter()
        for term in q_tf.keys():
            extension_terms = self.get_extension_terms(term)
            qf = q_tf[term]
            postings = self.get_posting(term)
            matching_term_list = [term] + extension_terms
            match_cnt = Counter()
            for matching_term in matching_term_list:
                for doc_id, cnt in self.get_posting(matching_term):
                    if matching_term == term:
                        factor = cnt
                    else:
                        factor = self.mapping_val
                    match_cnt[doc_id] += factor

            qdf = len(postings)
            for doc_id, cnt in match_cnt.items():
                tf = cnt
                dl = self.dl_d[doc_id]
                doc_score[doc_id] += self.scoring_fn(tf, qf, dl, qdf)

        return list(doc_score.items())


DocID = Union[int, str]
class BM25T_Retriever2(RetrieverIF):
    def __init__(
            self,
            get_posting: Callable[[str], List[Tuple[DocID, int]]],
            df,
            dl_d,
            scoring_fn,
            tokenize_fn,
            table: Dict[str, Dict[str, float]],
            stopwords,
    ):
        self.get_posting = get_posting
        self.scoring_fn = scoring_fn
        self.df = df
        self.tokenize_fn = tokenize_fn
        self.dl_d = dl_d
        self.extension_term_set_d: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for q_term, entries in table.items():
            d_term_score_pairs: List[Tuple[str, float]] = dict_to_tuple_list(entries)
            d_term_score_pairs.sort(key=get_second, reverse=True)
            self.extension_term_set_d[q_term] = d_term_score_pairs
        self.stopwords = set(stopwords)

    def _not_used_get_low_df_terms(self, q_terms: Iterable[str], n_limit=10) -> List[str]:
        candidates = []
        for t in q_terms:
            df = self.df[t]
            candidates.append((t, df))

        candidates.sort(key=get_second)
        return left(candidates)[:n_limit]

    def retrieve(self, query, n_retrieve=1000) -> List[Tuple[str, float]]:
        ret = self._retrieve_inner(query, n_retrieve)
        output: List[Tuple[str, float]] = []
        for doc_id, score in ret:
            if type(doc_id) != str:
                doc_id = str(doc_id)
            output.append((doc_id, score))
        return output

    def _retrieve_inner(self, query, n_retrieve=1000) -> List[Tuple[DocID, float]]:
        c_log.debug("Query: %s", query)
        q_tokens = self.tokenize_fn(query)
        q_tf = Counter(q_tokens)
        q_terms = list(q_tf.keys())

        q_terms = [term for term in q_terms if term not in self.stopwords]
        q_term_df_pairs = []
        for t in q_terms:
            df = self.df[t]
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

        doc_score: Dict[DocID, float] = Counter()
        for idx, (q_term, qterm_df) in enumerate(q_term_df_pairs):
            qf = q_tf[q_term]
            max_tf: Dict[str, float] = {}
            c_log.debug("Query term %s", q_term)
            c_log.debug("Request postings")

            if len(doc_score) >= n_retrieve:
                c_log.debug("Checking filtering")
                doc_score_pair_list = list(doc_score.items())
                doc_score_pair_list.sort(key=get_second, reverse=True)
                max_future_gain = sum(max_gain_per_term[idx:])
                _nth_doc, nth_doc_score = doc_score_pair_list[n_retrieve-1]
                only_check_known_docs = max_future_gain < nth_doc_score and False
            else:
                only_check_known_docs = False

            def get_posting_local(q_term):
                postings = self.get_posting(q_term)
                if only_check_known_docs:
                    target_doc_ids = list(doc_score.keys())
                    target_doc_ids.sort()
                    postings = self.join_postings(postings, target_doc_ids)
                return postings

            target_term_postings = get_posting_local(q_term)
            c_log.debug("Update counts")
            for doc_id, cnt in target_term_postings:
                max_tf[doc_id] = cnt

            target_term_posting_len = len(target_term_postings)

            total_posting_len = 0
            c_log.debug("Query term %s has %d extensions", q_term, len(self.extension_term_set_d[q_term]))
            for matching_term, match_score in self.extension_term_set_d[q_term]:
                c_log.debug("Request postings for %s", matching_term)
                postings = get_posting_local(matching_term)
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
            for doc_id, cnt in max_tf.items():
                tf = cnt
                dl = self.dl_d[doc_id]
                doc_score[doc_id] += self.scoring_fn(tf, qf, dl, qdf)
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
                    new_doc_score = Counter()
                    for doc_id, score in doc_score_pair_list:
                        if score >= drop_threshold:
                            new_doc_score[doc_id] = score
                        else:
                            break
                    c_log.debug("Keep %d docs from %d", len(new_doc_score), len(doc_score_pair_list))
                    doc_score = new_doc_score

        doc_score_pair_list: List[Tuple[DocID, float]] = list(doc_score.items())
        doc_score_pair_list.sort(key=get_second, reverse=True)
        return doc_score_pair_list

