import tensorflow as tf
from abc import abstractmethod, ABC
from collections import Counter, defaultdict
from typing import List, Iterable, Callable, Dict, Tuple, Set, TypeVar, Union

from adhoc.retriever_if import RetrieverIF
from adhoc.kn_tokenizer import KrovetzNLTKTokenizer
from list_lib import left
from misc_lib import get_second, dict_to_tuple_list
from typing import List, Iterable, Dict, Tuple, Set

from trainer_v2.chair_logging import c_log



DocID = Union[int, str]

class IndexReaderIF(ABC):
    @abstractmethod
    def get_df(self, term) -> int:
        pass

    @abstractmethod
    def get_dl(self, doc_id) -> int:
        pass

    @abstractmethod
    def get_postings(self, term) -> List[Tuple[DocID, int]]:
        pass


class IndexReaderPython(IndexReaderIF):
    def __init__(self, get_posting_fn, df, dl_d):
        self.get_posting = get_posting_fn
        self.df = df
        self.dl_d = dl_d


    def get_df(self, term) -> int:
        return self.df[term]

    def get_dl(self, doc_id) -> int:
        return self.dl_d[doc_id]

    def get_postings(self, term) -> List[Tuple[DocID, int]]:
        return self.get_posting(term)


class BM25T_Retriever(RetrieverIF):
    def __init__(
            self,
            index_reader: IndexReaderIF,
            scoring_fn,
            table: Dict[str, List[str]],
            mapping_val=0.1
    ):
        self.index_reader = index_reader
        self.scoring_fn = scoring_fn
        self.tokenizer = KrovetzNLTKTokenizer(False)
        self.tokenize_fn = self.tokenizer.tokenize_stem
        self.table: Dict[str, List[str]] = table
        self.mapping_val = mapping_val

    def get_extension_terms(self, term) -> List[str]:
        if term in self.table:
            return self.table[term]
        else:
            return []

    def retrieve(self, query, n_retrieve=1000) -> List[Tuple[str, float]]:
        q_tokens = self.tokenize_fn(query)
        q_tf = Counter(q_tokens)
        doc_score = Counter()
        for term in q_tf.keys():
            extension_terms = self.get_extension_terms(term)
            qf = q_tf[term]
            postings = self.index_reader.get_postings(term)
            matching_term_list = [term] + extension_terms
            match_cnt = Counter()
            for matching_term in matching_term_list:
                for doc_id, cnt in self.index_reader.get_postings(matching_term):
                    if matching_term == term:
                        factor = cnt
                    else:
                        factor = self.mapping_val
                    match_cnt[doc_id] += factor

            qdf = len(postings)
            for doc_id, cnt in match_cnt.items():
                tf = cnt
                dl = self.index_reader.get_dl(doc_id)
                doc_score[doc_id] += self.scoring_fn(tf, qf, dl, qdf)

        return list(doc_score.items())

class BM25T_Retriever2(RetrieverIF):
    def __init__(
            self,
            index_reader: IndexReaderIF,
            scoring_fn,
            tokenize_fn,
            table: Dict[str, Dict[str, float]],
            stopwords,
    ):
        self.index_reader = index_reader
        self.scoring_fn = scoring_fn
        self.tokenize_fn = tokenize_fn
        self.extension_term_set_d: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for q_term, entries in table.items():
            d_term_score_pairs: List[Tuple[str, float]] = dict_to_tuple_list(entries)
            d_term_score_pairs.sort(key=get_second, reverse=True)
            self.extension_term_set_d[q_term] = d_term_score_pairs
        self.stopwords = set(stopwords)

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

        doc_score: Dict[DocID, float] = Counter()
        for idx, (q_term, qterm_df) in enumerate(q_term_df_pairs):
            qf = q_tf[q_term]
            max_tf: Dict[str, float] = Counter()
            c_log.debug("Query term %s", q_term)

            if len(doc_score) > n_retrieve:
                doc_score_pair_list = list(doc_score.items())
                doc_score_pair_list.sort(key=get_second, reverse=True)
                max_future_gain = sum(max_gain_per_term[idx:])
                _nth_doc, nth_doc_score = doc_score_pair_list[n_retrieve-1]
                only_check_known_docs = max_future_gain < nth_doc_score and False
                c_log.debug(
                    "%s candidates is larger than %d, only_check_known_docs=%s",
                    len(doc_score), n_retrieve, str(only_check_known_docs))
            else:
                only_check_known_docs = False

            def get_posting_local(q_term):
                c_log.debug("Request postings")
                postings = self.index_reader.get_postings(q_term)
                if only_check_known_docs:
                    c_log.debug("Join postings")
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
            min_score = 1000 * 1000
            for doc_id, cnt in max_tf.items():
                tf = cnt
                dl = self.index_reader.get_dl(doc_id)
                doc_score[doc_id] += self.scoring_fn(tf, qf, dl, qdf)
                cur_score = doc_score[doc_id]
                min_score = min(min_score, cur_score)
            c_log.debug("Done score updates")

            # Drop not-promising docs
            # if len(doc_score) > n_retrieve:
            #     c_log.debug("Checking filtering")
            #     doc_score_pair_list = list(doc_score.items())
            #     doc_score_pair_list.sort(key=get_second, reverse=True)
            #     max_future_gain = sum(max_gain_per_term[idx+1:])
            #     _nth_doc, nth_doc_score = doc_score_pair_list[n_retrieve-1]
            #     drop_threshold: float = nth_doc_score - max_future_gain
            #     if drop_threshold > 0:
            #         new_doc_score: dict[DocID, float] = Counter()
            #         for doc_id, score in doc_score_pair_list:
            #             if score >= drop_threshold:
            #                 new_doc_score[doc_id] = score
            #             else:
            #                 break
            #         c_log.debug("Keep %d docs from %d", len(new_doc_score), len(doc_score_pair_list))
            #         doc_score = new_doc_score

        doc_score_pair_list: List[Tuple[DocID, float]] = list(doc_score.items())
        doc_score_pair_list.sort(key=get_second, reverse=True)
        return doc_score_pair_list

