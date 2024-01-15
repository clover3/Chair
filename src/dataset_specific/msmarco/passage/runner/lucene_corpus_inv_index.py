import sys
from collections import Counter
from typing import List, Iterable, Tuple, Dict


from omegaconf import OmegaConf

from adhoc.build_index import build_inverted_index_plus, save_inv_index_to_pickle
from cpath import output_path
from dataset_specific.msmarco.passage.tokenize_helper import iter_tokenized_corpus
from misc_lib import TELI
from misc_lib import path_join
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log

def build_inverted_index(
        tokenized_itr: Iterable[Tuple[str, List[str]]],
        num_docs=None,
        ):
    if num_docs is not None:
        itr = TELI(tokenized_itr, num_docs)
    else:
        itr = tokenized_itr
    inverted_index: Dict[str, List[Tuple[str, int]]] = {}
    df = Counter()
    dl = {}
    cdf = 0
    for doc_id, word_tokens in itr:
        count = Counter(word_tokens)
        for token, cnt in count.items():
            if token not in inverted_index:
                inverted_index[token] = []
            inverted_index[token].append((doc_id, cnt))
            df[token] += 1
        dl[doc_id] = sum(count.values())
        cdf += 1
    return {
        'df': df,
        "dl": dl,
        "cdf": cdf,
        'inverted_index': inverted_index,
    }


def common_index_preprocessing():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    print(conf)
    collection_size = 8841823
    save_path = path_join(output_path, "mmp", "passage_lucene", "all.tsv")
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
