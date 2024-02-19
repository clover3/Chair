import sys
from collections import Counter
from typing import List, Iterable, Tuple, Dict


from omegaconf import OmegaConf

from adhoc.build_index import build_inverted_index_plus, save_inv_index_to_pickle
from cpath import output_path
from dataset_specific.msmarco.passage.runner.lucene_corpus_inv_index import build_inverted_index
from dataset_specific.msmarco.passage.tokenize_helper import iter_tokenized_corpus
from misc_lib import TELI
from misc_lib import path_join
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log

def common_index_preprocessing():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    print(conf)
    collection_size = 8841823
    save_path = path_join(output_path, "mmp", "passage_lucene_k", "all.tsv")
    itr = iter_tokenized_corpus(save_path)
    tokenized_itr: Iterable[Tuple[str, List[str]]] = itr
    if collection_size is not None:
        itr = TELI(tokenized_itr, collection_size)
    else:
        itr = tokenized_itr
    df = Counter()
    cdf = 0
    for doc_id, word_tokens in itr:
        count = Counter(word_tokens)
        for token, cnt in count.items():
            df[token] += 1
            if df[token] > collection_size:
                print(f"Term {token} exceeded the collection size")
        cdf += 1


def main():
    with JobContext("inv_index_building"):
        common_index_preprocessing()


if __name__ == "__main__":
    main()
