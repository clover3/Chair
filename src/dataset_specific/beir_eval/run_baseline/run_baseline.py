import sys

from adhoc.bm25_retriever import BM25RetrieverKNTokenize, build_bm25_scoring_fn
from adhoc.retriever_if import RetrieverIF
from cache import load_pickle_from
from dataset_specific.beir_eval.beir_common import beir_dataset_list_A
from dataset_specific.beir_eval.path_helper import get_beir_inv_index_path, get_beir_df_path, get_beir_dl_path
from dataset_specific.beir_eval.run_helper import run_retrieval_and_eval_on_beir
from misc_lib import average

from trainer_v2.chair_logging import c_log


def load_bm25_retriever(dataset) -> BM25RetrieverKNTokenize:
    inv_index = load_pickle_from(get_beir_inv_index_path(dataset))
    df = load_pickle_from(get_beir_df_path(dataset))
    dl = load_pickle_from(get_beir_dl_path(dataset))
    cdf = len(dl)
    avdl = average(dl.values())
    scoring_fn = build_bm25_scoring_fn(cdf, avdl)
    bm25_retriever = BM25RetrieverKNTokenize(inv_index, df, dl, scoring_fn)
    return bm25_retriever


def run_bm25(dataset):
    c_log.info(f"run_bm25 on {dataset}")
    method = "bm25"
    max_doc_per_list = 1000
    split = "test"
    retriever: RetrieverIF = load_bm25_retriever(dataset)

    run_retrieval_and_eval_on_beir(dataset, split, method, retriever, max_doc_per_list)


def main():
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
        run_bm25(dataset)
    else:
        for dataset in beir_dataset_list_A:
            run_bm25(dataset)


if __name__ == "__main__":
    main()
