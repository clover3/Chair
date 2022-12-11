from collections import Counter

from adhoc.bm25_class import BM25
from adhoc.clueweb12_B13_termstat import cdf, load_clueweb12_B13_termstat_stemmed
from adhoc.kn_tokenizer import KrovetzNLTKTokenizer
from contradiction.medical_claims.retrieval.defs import BioClaimRetrievalSystem
from typing import List, Iterable, Callable, Dict, Tuple, Set

from misc_lib import Averager


class BM25Clueweb(BioClaimRetrievalSystem):
    def __init__(self):
        tf, df = load_clueweb12_B13_termstat_stemmed()
        self.bm25 = BM25(df, avdl=11.7, num_doc=cdf, k1=0.00001, k2=100, b=0.5,
                         drop_stopwords=True)

    def score(self, question: str, claim: str) -> float:
        return self.bm25.score(question, claim)


def build_stats(text_list: List[str]):
    cdf = len(text_list)
    tokenizer = KrovetzNLTKTokenizer()
    avg_dl = Averager()
    df = Counter()
    for text in text_list:
        tokens = tokenizer.tokenize_stem(text)
        avg_dl.append(len(tokens))
        df.update(set(tokens))

    return df, cdf, avg_dl.get_average()


class BM25BioClaim(BioClaimRetrievalSystem):
    def __init__(self, texts):
        df, cdf, avdl = build_stats(texts)
        self.bm25 = BM25(df, avdl=avdl, num_doc=cdf, k1=0.00001, k2=100, b=0.5,
                         drop_stopwords=True)

    def score(self, question: str, claim: str) -> float:
        return self.bm25.score(question, claim)
