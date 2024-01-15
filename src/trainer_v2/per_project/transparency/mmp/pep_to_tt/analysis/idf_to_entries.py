import math
import sys
from collections import Counter

from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from table_lib import tsv_iter


def main():
    def is_number_like(term: str) -> bool:
        for i in range(10):
            if str(i) in term:
                return True

        return False



    counter = Counter()
    for q_term, d_term, score in tsv_iter(sys.argv[1]):
        counter[q_term] += 1

    cdf, df_d = load_msmarco_passage_term_stat()

    def get_idf(df):
        return math.log((cdf-df+0.5)/(df + 0.5))

    for term, cnt in counter.most_common(100):
        df = df_d[term]
        print(term, cnt, df, get_idf(df))




if __name__ == "__main__":
    main()
