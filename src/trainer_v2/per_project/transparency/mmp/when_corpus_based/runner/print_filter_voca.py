from krovetzstemmer import Stemmer
from cpath import output_path
from misc_lib import path_join
from tab_print import print_table

from trainer_v2.per_project.transparency.mmp.when_corpus_based.when_bm25t import load_align_weights


def main():
    global_align_path = path_join(
        output_path, "msmarco", "passage", "when_global_align_filter")

    f = open(global_align_path, "w")
    itr = load_align_weights()
    stemmer = Stemmer()
    min_tf = 10
    l = []
    table = []
    for t in itr:
        if t.n_appear >= min_tf and t.score > 0.01:
            table.append([t.word, t.score, t.n_appear, t.n_pos_appear])
            word = stemmer(t.word)
            l.append(word)
    print("Selected {}".format(len(l)))
    voca = {}
    for idx, term in enumerate(l):
        voca[term] = idx + 1

    print_table(table)



if __name__ == "__main__":
    main()