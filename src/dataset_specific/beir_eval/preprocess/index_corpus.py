import pickle

from adhoc.kn_tokenizer import KrovetzNLTKTokenizer
from dataset_specific.beir_eval.beir_common import load_beir_dataset, beir_dataset_list_not_large
from adhoc.build_index import build_inverted_index, count_df_from_tokenized, count_dl_from_tokenized
from dataset_specific.beir_eval.path_helper import get_beir_inv_index_path, get_beir_df_path, get_beir_dl_path
from trainer_v2.chair_logging import c_log
from typing import List, Iterable, Callable, Dict, Tuple, Set


def common_index_preprocessing(dataset):
    tokenizer = KrovetzNLTKTokenizer()
    tokenize_fn = tokenizer.tokenize_stem
    corpus, queries, qrels = load_beir_dataset(dataset, "test")

    def iter_tokenized():
        for doc_id, doc in corpus.items():
            doc_text = doc['text']
            yield doc_id, tokenize_fn(doc_text)

    corpus_tokenized: List[Tuple[str, List[str]]] = list(iter_tokenized())

    c_log.info("Count df")
    df = count_df_from_tokenized(corpus_tokenized)
    pickle.dump(df, open(get_beir_df_path(dataset), "wb"))
    c_log.info("Count dl")
    dl = count_dl_from_tokenized(corpus_tokenized)
    pickle.dump(dl, open(get_beir_dl_path(dataset), "wb"))
    ignore_voca = set()
    c_log.info("Building inverted index")
    inverted_index = build_inverted_index(
        corpus_tokenized,
        ignore_voca,
        num_docs=len(corpus)
    )
    pickle.dump(inverted_index, open(get_beir_inv_index_path(dataset), "wb"))


def main():
    for dataset in beir_dataset_list_not_large:
        c_log.info(f"Working on {dataset}")
        common_index_preprocessing(dataset)


if __name__ == "__main__":
    main()
