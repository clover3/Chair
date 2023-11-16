from adhoc.bm25_class import BM25
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat


def get_bm25_mmp_25_01_01() -> BM25:
    cdf, df = load_msmarco_passage_term_stat()
    bm25 = BM25(df, cdf, 25, k1=0.1, k2=0, b=0.1)
    return bm25



def main():
    cdf, df = load_msmarco_passage_term_stat()
    bm25 = BM25(df, cdf, 25, k1=0.1, k2=0, b=0.1)
    print("CDF", cdf)

    for t in bm25.core.df:
        qtf = bm25.term_idf_factor(t)
        if not qtf > 0:
            print(t, qtf, df[t])


if __name__ == "__main__":
    main()
