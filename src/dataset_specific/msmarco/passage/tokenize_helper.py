from misc_lib import TimeEstimator
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator


def corpus_tokenize(itr: Iterable[Tuple[str, str]], tokenize_fn, save_path, collection_size):
    f = open(save_path, "w")
    ticker = TimeEstimator(collection_size)
    for doc_id, text in itr:
        tokens = tokenize_fn(text)
        ticker.tick()
        tokens_str = " ".join(tokens)
        f.write(f"{doc_id}\t{tokens_str}\n")


def iter_tokenized_corpus(save_path):
    f = open(save_path, "r")
    for line in f:
        doc_id, doc_tokens_str = line.split("\t")
        doc_tokens = doc_tokens_str.split()
        yield doc_id, doc_tokens
