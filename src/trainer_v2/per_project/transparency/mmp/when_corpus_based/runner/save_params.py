import csv
import sys

from krovetzstemmer import Stemmer

from cpath import output_path
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.bm25t import GlobalAlign
from trainer_v2.per_project.transparency.mmp.when_corpus_based.when_bm25t import load_global_aligns
from typing import List, Iterable, Callable, Dict, Tuple, Set


def build_table(global_align_itr: Iterable[GlobalAlign]) -> Dict[str, float]:
    stemmer = Stemmer()
    min_tf = 10
    out_mapping: Dict[str, float] = {}
    n_all = 0
    for t in global_align_itr:
        n_all += 1
        rate = t.n_pos_appear / t.n_appear
        if t.n_appear >= min_tf and t.score > 0.01 and rate > 0.6:
            word = stemmer(t.word)
            out_mapping[word] = t.score
    print("Selected {} from {} items".format(len(out_mapping), n_all))
    return out_mapping


def save_param(d: Dict,  name):
    save_path = path_join(
        output_path, "msmarco", "passage", "when_trained_saved", name)

    output_row = [(k, v) for k, v in d.items()]

    with open(save_path, "w", newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(output_row)


def main():
    global_align_path = sys.argv[1]
    save_name = sys.argv[2]
    global_align_itr: Iterable[GlobalAlign] = load_global_aligns(global_align_path)
    mapping = build_table(global_align_itr)
    save_param(mapping, save_name)


if __name__ == "__main__":
    main()