import random
from collections import Counter
from dataclasses import dataclass
from typing import List, Iterable, Callable, Dict, Tuple, Set

from cpath import output_path
from misc_lib import path_join

from krovetzstemmer import Stemmer

from dataset_specific.msmarco.passage.passage_resource_loader import tsv_iter


@dataclass
class Entry:
    q_term:str
    d_term: str
    acc_score: float
    non_zero_tf_of_pair: int
    corpus_tf_q_term_id:int
    corpus_tf_d_term_id: int


def main():
    stemmer = Stemmer()

    candidate_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "global_pair_on_when.tsv")
    save_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "candidate1.tsv")
    rows = tsv_iter(candidate_path)

    def compute_metric(e: Entry):
        return e.acc_score / e.corpus_tf_d_term_id

    def parse_row(row):
        q_term, d_term, acc_score, non_zero_tf_of_pair, corpus_tf_q_term_id, corpus_tf_d_term_id = row
        # Exclude if q_term and d_term are equal when stemmed
        entry = Entry(
            q_term, d_term, float(acc_score),
            int(non_zero_tf_of_pair), int(corpus_tf_q_term_id), int(corpus_tf_d_term_id)
        )
        return entry

    def is_same_stemmed(entry: Entry):
        q_stemmed = stemmer.stem(entry.q_term)
        d_stemmed = stemmer.stem(entry.d_term)

        if q_stemmed == d_stemmed:
            # counter["stem equal"] += 1
            # if entry.q_term == entry.d_term:
            #     counter["term equal"] += 1
            return True

    def frequent(entry):
        return entry.corpus_tf_d_term_id > 10

    def not_same(e):
        return not is_same_stemmed(e)

    def not_subtail(e):
        return e.q_term[0] != "#" and e.d_term[0] != '#'

    items = map(parse_row, rows)
    items = filter(not_same, items)
    items = filter(not_subtail, items)
    items = filter(frequent, items)

    items = list(items)
    items.sort(key=compute_metric, reverse=True)

    n_top = 3000
    n_middle = 4000
    n_bottom = 3000

    top_items = items[:n_top]
    bottom_items = items[-n_bottom:]

    middle_items_all = items[n_top: -n_bottom]
    random.shuffle(middle_items_all)
    middle_items = middle_items_all[:n_middle]

    selected: List[Entry] = top_items + middle_items + bottom_items

    save_f = open(save_path, "w")
    for e in selected:
        row = e.q_term, e.d_term
        out_line = "\t".join(map(str, row))
        save_f.write(out_line + "\n")





if __name__ == "__main__":
    main()