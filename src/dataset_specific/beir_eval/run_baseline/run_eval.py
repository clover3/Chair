import json
import sys

from beir.retrieval.evaluation import EvaluateRetrieval

from adhoc.bm25_retriever import BM25RetrieverKNTokenize, build_bm25_scoring_fn, RetrieverIF
from cache import load_pickle_from
from dataset_specific.beir_eval.beir_common import load_beir_dataset, beir_dataset_list_not_large
from dataset_specific.beir_eval.path_helper import get_beir_inv_index_path, get_beir_df_path, get_beir_dl_path, \
    get_json_qres_save_path
from misc_lib import average, get_second, TimeEstimator
from typing import List, Iterable, Callable, Dict, Tuple, Set

from trainer_v2.chair_logging import c_log


def run_eval(dataset, method):
    split = "test"
    c_log.info(f"Loading dataset")
    run_name = f"{dataset}_{method}"
    _, queries, qrels = load_beir_dataset(dataset, split)
    json_qres_save_path = get_json_qres_save_path(run_name)

    output = json.load(open(json_qres_save_path, "r"))
    ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, output, [1, 10, 100, 1000])
    results2 = EvaluateRetrieval.evaluate_custom(qrels, output, [1, 10, 100, 1000], metric="r_cap")
    eval_res = {
        "NDCG@10": ndcg["NDCG@10"],
        "Recall@100": recall["Recall@100"],
        "R_cap@100": results2["R_cap@100"]
    }
    print(eval_res)


def main():
    dataset = sys.argv[1]
    method = "bm25"
    run_eval(dataset, method)


if __name__ == "__main__":
    main()
