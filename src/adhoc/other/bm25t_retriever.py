from abc import ABC, abstractmethod
from collections import Counter
from adhoc.bm25_retriever import RetrieverIF
from adhoc.kn_tokenizer import KrovetzNLTKTokenizer
from list_lib import left
from misc_lib import get_second
from typing import List, Iterable, Callable, Dict, Tuple, Set


class BM25T_Retriever(RetrieverIF):
    def __init__(
            self, inv_index, df, dl_d,
            scoring_fn,
            table: Dict[str, List[str]]):
        self.inv_index = inv_index
        self.scoring_fn = scoring_fn
        self.df = df
        self.tokenizer = KrovetzNLTKTokenizer(False)
        self.tokenize_fn = self.tokenizer.tokenize_stem
        self.dl_d = dl_d
        self.table: Dict[str, List[str]] = table

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
                        factor = 0.1
                    match_cnt[doc_id] += factor

            qdf = len(postings)
            for doc_id, cnt in match_cnt.items():
                tf = cnt
                dl = self.dl_d[doc_id]
                doc_score[doc_id] += self.scoring_fn(tf, qf, dl, qdf)

        return list(doc_score.items())
