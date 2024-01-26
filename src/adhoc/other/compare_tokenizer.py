import nltk

from adhoc.other.bm25_retriever_helper import get_tokenize_fn

from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator
import sys
from omegaconf import OmegaConf

from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lmap, left, right
from misc_lib import select_third_fourth
from table_lib import tsv_iter

def is_all_equal(l: list[int]):
    # return if all elements are equal
    if len(l) == 0:
        return True

    # Get the first element of the list
    first_element = l[0]

    # Iterate through the list starting from the second element
    for element in l[1:]:
        if element != first_element:
            return False

    # If we reached this point, all elements are equal
    return True


def main():
    tokenizer = get_tokenizer()

    tokenizer_d = {
        "NLTK": nltk.tokenize.word_tokenize,
        "Bert": tokenizer.basic_tokenizer.tokenize
    }

    quad_tsv_path = "data/msmarco/dev_sample1000/corpus.tsv"
    qd_iter: Iterable[Tuple[str, str]] = select_third_fourth(tsv_iter(quad_tsv_path))
    n_print = 1000
    query_path = "data/msmarco/dev_sample1000/queries.tsv"
    queries = tsv_iter(query_path)
    text_itr = right(qd_iter)

    for t in text_itr:
        tokens_d = {}
        for name, fn in tokenizer_d.items():
            tokens = fn(t)
            tokens_d[name] = tokens

        len_equal = is_all_equal(lmap(len, tokens_d.values()))

        if len(tokens_d["NLTK"]) > len(tokens_d["Bert"]) and "cannot" not in t:
            print("Text", t)
            for name in tokenizer_d:
                tokens = tokens_d[name]
                print("{}\t{}\t{}".format(name, len(tokens), " ".join(tokens)))
            print()
            n_print = n_print - 1

            if n_print == 0:
                break


if __name__ == "__main__":
    main()
