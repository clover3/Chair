import pickle

import nltk

from adhoc.build_index import build_inverted_index_plus
from dataset_specific.msmarco.passage.doc_indexing.index_path_helper import get_mmp_df_path, get_mmp_dl_path, \
    get_mmp_inv_index_path, get_bm25_no_stem_resource_path_helper
from dataset_specific.msmarco.passage.passage_resource_loader import load_msmarco_collection
from misc_lib import TELI
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from typing import List, Iterable, Callable, Dict, Tuple, Set


def common_index_preprocessing():
    index_name = "no_stem"
    tokenize_fn = nltk.tokenize.word_tokenize
    itr = load_msmarco_collection()
    collection_size = 8841823
    itr = TELI(itr, collection_size)

    def iter_tokenized():
        for doc_id, doc_text in itr:
            yield doc_id, tokenize_fn(doc_text)

    corpus_tokenized: List[Tuple[str, List[str]]] = list(iter_tokenized())
    ignore_voca = set()
    c_log.info("Building inverted index")
    outputs = build_inverted_index_plus(
        corpus_tokenized,
        ignore_voca,
        num_docs=collection_size
    )

    inverted_index = outputs["inverted_index"]
    dl = outputs["dl"]
    df = outputs["df"]

    # conf = get_bm25_no_stem_resource_path_helper()
    # conf.inv_index_path
    pickle.dump(df, open(get_mmp_df_path(index_name), "wb"))
    pickle.dump(dl, open(get_mmp_dl_path(index_name), "wb"))
    pickle.dump(inverted_index, open(get_mmp_inv_index_path(index_name), "wb"))



def main():
    with JobContext("inv_index_building"):
        common_index_preprocessing()


if __name__ == "__main__":
    main()