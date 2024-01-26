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
    corpus_tokenized: Iterable[Tuple[str, List[str]]] = itr
    c_log.info("Building inverted index")
    outputs = build_inverted_index(
        corpus_tokenized,
        collection_size,
    )
    c_log.info("Building inverted Done")

    save_inv_index_to_pickle(conf, outputs)


def main():
    with JobContext("inv_index_building"):
        common_index_preprocessing()


if __name__ == "__main__":
    main()
