from collections import Counter

from krovetzstemmer import Stemmer
from cpath import output_path
from misc_lib import path_join
from tab_print import print_table

from trainer_v2.per_project.transparency.mmp.when_corpus_based.when_bm25t import load_align_weights


def main():
    itr = load_align_weights()
    stemmer = Stemmer()
    min_tf = 10
    counter = Counter()
    l = []
    table = []
    for t in itr:
        f_enough_seen = t.n_appear >= min_tf
        f_score_high = t.score > 0.01
        if f_score_high:
            counter['f_score_high'] += 1
        if f_enough_seen:
            counter['f_enough_seen'] += 1
        counter['f_all'] += 1

        if f_enough_seen and f_score_high:
            counter['f_selected'] += 1
            table.append([t.word, t.score, t.n_appear, t.n_pos_appear])
            word = stemmer(t.word)
            l.append(word)
    print("Selected {}".format(len(l)))
    voca = {}
    for idx, term in enumerate(l):
        voca[term] = idx + 1

    print_table(table)
    print(counter)


if __name__ == "__main__":
    main()