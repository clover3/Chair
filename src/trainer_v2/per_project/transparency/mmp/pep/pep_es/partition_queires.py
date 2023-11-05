import random
import sys
from typing import List, Iterable, Callable, Dict, Tuple, Set

from transformers import AutoTokenizer

from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.passage.path_helper import iter_train_triples_partition
from dataset_specific.msmarco.passage.processed_resource_loader import get_partitioned_query_path
from misc_lib import TELI


def iter_queries(part_no):
    for query, _d1, _d2 in iter_train_triples_partition(part_no):
        yield query


def get_random_split_location_on_sp_tokens(tokens) -> Tuple[int, int]:
    st = random.randint(0, len(tokens) - 1)
    ed = random.randint(st+1, len(tokens))
    return st, ed


def main():
    random.seed(0)
    tokenizer = get_tokenizer()
    sp_tokenize = tokenizer.basic_tokenizer.tokenize
    part_no = int(sys.argv[1])
    f = open(get_partitioned_query_path(part_no), "w", encoding="utf-8")

    for query_text in TELI(iter_queries(part_no), 1000000):
        sp_tokens: list[str] = sp_tokenize(query_text)
        st, ed = get_random_split_location_on_sp_tokens(sp_tokens)
        row = " ".join(sp_tokens), str(st), str(ed)
        f.write("\t".join(row)+"\n")


if __name__ == "__main__":
    main()