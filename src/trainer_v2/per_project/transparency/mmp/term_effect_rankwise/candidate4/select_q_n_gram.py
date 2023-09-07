from collections import Counter
from typing import List, Iterable, Callable, Dict, Tuple, Set

from dataset_specific.msmarco.passage.passage_resource_loader import enum_queries
from trainer_v2.per_project.transparency.misc_common import save_tsv
from cpath import output_path
from misc_lib import path_join


def enum_n_gram(q_tokens: List[str], n_gram_width: int) -> Iterable[List[str]]:
    for j in range(len(q_tokens) + 1 - n_gram_width):
        n_gram = q_tokens[j:j + n_gram_width]
        yield n_gram


def main():
    split = "train"
    reader = enum_queries(split)
    n_gram_width = 2
    counter: Counter[str, int] = Counter()
    for idx, row in enumerate(reader):
        qid, q_text = row
        q_tokens = q_text.lower().split()

        for j in range(len(q_tokens) + 1 - n_gram_width):
            n_gram = q_tokens[j:j+n_gram_width]
            assert len(n_gram) == n_gram_width
            counter[" ".join(n_gram)] += 1

    def output_iter():
        for n_gram, cnt in counter.most_common(100 * 1000):
            row = [n_gram, cnt]
            yield row

    save_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "cand4_freq_q_terms.tsv")
    save_tsv(output_iter(), save_path)


if __name__ == "__main__":
    main()