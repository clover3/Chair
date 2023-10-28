from typing import List, Iterable, Tuple

from krovetzstemmer import Stemmer

from adhoc.build_index import build_inverted_index_plus, save_inv_index_to_pickle
from cpath import output_path
from dataset_specific.msmarco.passage.doc_indexing.index_path_helper import \
    get_bm25_bt3_resource_path_helper
from dataset_specific.msmarco.passage.runner.bt2_inv_index import iter_bert_tokenized_merged
from misc_lib import TELI
from misc_lib import path_join
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log


def common_index_preprocessing():
    conf = get_bm25_bt3_resource_path_helper()
    stemmer = Stemmer()

    def stem_tokens(pair):
        doc_id, tokens = pair
        st_tokens = [stemmer.stem(t) for t in tokens]
        return doc_id, st_tokens

    collection_size = 8841823
    itr = iter_bert_tokenized_merged()
    itr = map(stem_tokens, itr)
    itr = TELI(itr, collection_size)
    corpus_tokenized: Iterable[Tuple[str, List[str]]] = itr
    ignore_voca = set()
    c_log.info("Building inverted index")
    outputs = build_inverted_index_plus(
        corpus_tokenized,
        ignore_voca,
        num_docs=collection_size
    )
    c_log.info("Building inverted Done")
    save_inv_index_to_pickle(conf, outputs)


def main():
    with JobContext("inv_index_building_bt3"):
        common_index_preprocessing()


if __name__ == "__main__":
    main()