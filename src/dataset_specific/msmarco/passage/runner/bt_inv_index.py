import pickle
from typing import List, Iterable, Tuple

from adhoc.build_index import build_inverted_index_plus
from cpath import output_path
from dataset_specific.msmarco.passage.doc_indexing.index_path_helper import get_bm25_bert_tokenized_resource_path_helper
from misc_lib import TELI
from misc_lib import path_join
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log


def iter_bert_tokenized_corpus():
    save_path = path_join(output_path, "msmarco", "passage_bert_tokenized.tsv")
    f = open(save_path, "r")
    for line in f:
        doc_id, doc_tokens_str = line.split("\t")
        doc_tokens = doc_tokens_str.split()
        yield doc_id, doc_tokens


def common_index_preprocessing():
    collection_size = 8841823
    itr = iter_bert_tokenized_corpus()
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

    inverted_index = outputs["inverted_index"]
    dl = outputs["dl"]
    df = outputs["df"]

    conf = get_bm25_bert_tokenized_resource_path_helper()

    c_log.info("Saving df")
    pickle.dump(df, open(conf.df_path, "wb"))

    c_log.info("Saving dl")
    pickle.dump(dl, open(conf.dl_path, "wb"))

    c_log.info("Saving inv_index")
    pickle.dump(inverted_index, open(conf.inv_index_path, "wb"))


def main():
    with JobContext("inv_index_building"):
        common_index_preprocessing()


if __name__ == "__main__":
    main()