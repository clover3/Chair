from math import log

from krovetzstemmer import Stemmer

from misc_lib import get_second
from trainer_v2.per_project.transparency.mmp.when_corpus_based.when_bm25t import load_align_weights


def score_candidates():
    itr = load_align_weights()
    stemmer = Stemmer()
    min_tf = 10
    ctf = 1e6
    l = []
    max_freq = 96135
    for t in itr:
        if t.n_appear > min_tf:
            n_neg = t.n_appear - t.n_pos_appear
            log_pos = log(t.n_pos_appear + 1 / max_freq)
            log_neg = log(n_neg + 1 / max_freq)
            log_odd = log_pos - log_neg
            l.append((t, log_odd))

    l.sort(key=get_second, reverse=True)
    for t, log_odd in l:
        print(t, log_odd)


def main():
    score_candidates()


if __name__ == "__main__":
    main()