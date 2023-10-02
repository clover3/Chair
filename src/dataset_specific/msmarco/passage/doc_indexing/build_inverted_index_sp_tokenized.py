import pickle

from krovetzstemmer import Stemmer

from adhoc.build_index import build_inverted_index_plus
from dataset_specific.msmarco.passage.doc_indexing.index_path_helper import get_mmp_df_path, get_mmp_dl_path, \
    get_mmp_inv_index_path, get_bm25_no_stem_resource_path_helper, get_bm25_sp_stem_resource_path_helper
from dataset_specific.msmarco.passage.passage_resource_loader import load_msmarco_collection
from misc_lib import TELI
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from typing import List, Tuple


def common_index_preprocessing():
    itr = load_msmarco_collection()
    collection_size = 8841823
    itr = TELI(itr, collection_size)
    stemmer = Stemmer()

    def apply_stem(tokens):
        stemmed_tokens = []
        for t in tokens:
            try:
                stemmed_tokens.append(stemmer.stem(t))
            except:
                pass
        return stemmed_tokens

    def iter_tokenized():
        for doc_id, doc_text in itr:
            tokens = doc_text.split()
            yield doc_id, apply_stem(tokens)

    corpus_tokenized: List[Tuple[str, List[str]]] = iter_tokenized()
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

    conf = get_bm25_sp_stem_resource_path_helper()

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