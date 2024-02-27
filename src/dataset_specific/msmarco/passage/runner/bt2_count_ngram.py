from typing import List, Iterable, Tuple

from nltk import ngrams

from adhoc.build_index import count_dl_df, save_df_dl
from dataset_specific.msmarco.passage.doc_indexing.index_path_helper import \
    get_bm25_bt2_resource_path_helper
from dataset_specific.msmarco.passage.runner.bt2_inv_index import iter_bert_tokenized_merged
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from typing import List, Iterable, Tuple

from adhoc.build_index import count_dl_df, save_df_dl
from dataset_specific.msmarco.passage.doc_indexing.index_path_helper import \
    get_bm25_bt2_resource_path_helper
from dataset_specific.msmarco.passage.runner.bt2_inv_index import iter_bert_tokenized_merged
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
import os
import pathlib
import pickle
from collections import Counter
from typing import Dict, List, Tuple
from typing import List, Iterable, Callable, Dict, Tuple, Set

from misc_lib import TELI
from trainer_v2.chair_logging import c_log



def count_ngram(
        tokenized_itr: Iterable[Tuple[str, List[str]]],
        ngram_range,
        num_docs=None):
    if num_docs is not None:
        itr = TELI(tokenized_itr, num_docs)
    else:
        itr = tokenized_itr
    counter_d = {n: Counter() for n in ngram_range}

    for doc_id, word_tokens in itr:
        for n in ngram_range:
            for tokens in ngrams(word_tokens, n):
                rep = " ".join(tokens)
                counter_d[n][rep] += 1

    return counter_d


def count_save_ngram():
    conf = get_bm25_bt2_resource_path_helper()
    dir_maybe = os.path.dirname(conf.df_path)
    collection_size = 8841823
    itr = iter_bert_tokenized_merged()
    corpus_tokenized: Iterable[Tuple[str, List[str]]] = itr
    ngram_range = [1, 3]
    print("ngram_range", ngram_range)
    c_log.info("Counting ngram tf")
    outputs = count_ngram(
        corpus_tokenized,
        ngram_range,
        num_docs=collection_size
    )

    for i in ngram_range:
        save_path = os.path.join(dir_maybe, "ngram_count_{}".format(i))
        pickle.dump(outputs[i], open(save_path, "wb"))

    c_log.info("Done")


def main():
    with JobContext("df_counting"):
        count_save_ngram()


if __name__ == "__main__":
    main()
