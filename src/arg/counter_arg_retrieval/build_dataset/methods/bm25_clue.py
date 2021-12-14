from arg.bm25 import BM25
from arg.clueweb12_B13_termstat import load_clueweb12_B13_termstat_stemmed_from_pickle, cdf


def build_bm25():
    avdl = 10
    tf, df = load_clueweb12_B13_termstat_stemmed_from_pickle()
    return BM25(df, avdl=avdl, num_doc=cdf, k1=0.001, k2=100, b=0.5)