import sys

from adhoc.bm25_retriever import BM25Retriever, build_bm25_scoring_fn, RetrieverIF
from adhoc.other.bm25t_retriever import BM25T_Retriever
from cache import load_pickle_from
from dataset_specific.beir_eval.beir_common import beir_dataset_list_not_large
from dataset_specific.beir_eval.path_helper import get_beir_inv_index_path, get_beir_df_path, get_beir_dl_path
from dataset_specific.beir_eval.run_helper import run_retrieval_and_eval
from misc_lib import average

from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bm25t_helper import load_binary_mapping_from_align_scores
from typing import List, Iterable, Callable, Dict, Tuple, Set


def load_bm25t_retriever(dataset, table_path) -> BM25T_Retriever:
    inv_index = load_pickle_from(get_beir_inv_index_path(dataset))
    df = load_pickle_from(get_beir_df_path(dataset))
    dl = load_pickle_from(get_beir_dl_path(dataset))
    cdf = len(dl)
    avdl = average(dl.values())
    scoring_fn = build_bm25_scoring_fn(cdf, avdl)

    table: Dict[str, List[str]] = load_binary_mapping_from_align_scores(table_path, 0.1)
    retriever = BM25T_Retriever(inv_index, df, dl, scoring_fn, table)
    return retriever


def run_bm25t(dataset, table_name, table_path):
    method_name = "bm25t_" + table_name
    c_log.info(f"run {method_name} on {dataset}")
    max_doc_per_list = 1000
    split = "test"
    retriever: RetrieverIF = load_bm25t_retriever(dataset, table_path)
    run_retrieval_and_eval(dataset, split, method_name, retriever, max_doc_per_list)


def main():
    dataset = sys.argv[1]
    table_name = sys.argv[2]
    table_path = sys.argv[3]
    run_bm25t(dataset, table_name, table_path)


if __name__ == "__main__":
    main()
