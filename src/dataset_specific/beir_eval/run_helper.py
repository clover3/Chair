import json
from typing import List, Tuple

from beir.retrieval.evaluation import EvaluateRetrieval

from adhoc.bm25_retriever import RetrieverIF
from dataset_specific.beir_eval.beir_common import load_beir_dataset
from dataset_specific.beir_eval.path_helper import get_json_qres_save_path
from misc_lib import TimeEstimator, get_second
from trainer_v2.chair_logging import c_log


def save_json_qres(run_name, output):
    json_qres_save_path = get_json_qres_save_path(run_name)
    json.dump(output, open(json_qres_save_path, "w"))


def run_retrieval(retriever: RetrieverIF, queries, max_doc_per_list):
    ticker = TimeEstimator(len(queries))
    output = {}
    for qid, query_text in queries.items():
        res: List[Tuple[str, float]] = retriever.retrieve(query_text)
        res.sort(key=get_second, reverse=True)

        per_query_res = {}
        for doc_id, score in res[:max_doc_per_list]:
            per_query_res[doc_id] = score

        output[qid] = per_query_res
        ticker.tick()
    return output


def run_retrieval_and_eval(dataset, split, method, retriever, max_doc_per_list):
    c_log.info(f"Loading dataset")
    _, queries, qrels = load_beir_dataset(dataset, split)
    c_log.info(f"run_retrieval")
    output = run_retrieval(retriever, queries, max_doc_per_list)
    run_name = f"{dataset}_{method}"
    save_json_qres(run_name, output)
    ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, output, [1, 10, 100, 1000])
    results2 = EvaluateRetrieval.evaluate_custom(qrels, output, [1, 10, 100, 1000], metric="r_cap")
    eval_res = {
        "NDCG@10": ndcg["NDCG@10"],
        "Recall@100": recall["Recall@100"],
        "R_cap@100": results2["R_cap@100"]
    }
    print(eval_res)