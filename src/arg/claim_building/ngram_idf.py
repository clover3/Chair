import sys

from arg.claim_building.count_ngram import load_n_gram_from_pickle, is_single_char_n_gram
from arg.claim_building.ngram_growth import df_to_idf
from arg.clueweb12_B13_termstat import load_subword_term_stat
from cache import load_from_pickle
from misc_lib import average


def show(n):
    topic = "abortion"
    count = load_n_gram_from_pickle(topic, n)
    clueweb_tf, clueweb_df = load_subword_term_stat()
    clueweb_idf = df_to_idf(clueweb_df)
    c_tf, nc_tf = load_from_pickle("abortion_clm")


    avg_idf = average(list(clueweb_idf.values()))

    def get_idf(t):
        if t in clueweb_idf:
            return clueweb_idf[t]
        else:
            return avg_idf


    l = list(count.items())
    skip_count = 0
    l.sort(key=lambda x:x[1], reverse=True)
    for n_gram, cnt in l[:1000]:
        if is_single_char_n_gram(n_gram):
            skip_count += 1
        else:
            idf_sum = sum([get_idf(t) for t in n_gram])
            print("{} {}".format(n_gram, cnt) + " {0:.2f} {1:.2f} ".format(idf_sum, cnt * idf_sum))

    print("Skip", skip_count)




if __name__ == "__main__":
    n = int(sys.argv[1])
    show(n)
