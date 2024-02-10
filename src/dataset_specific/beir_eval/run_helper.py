from beir.retrieval.evaluation import EvaluateRetrieval

from adhoc.adhoc_retrieval import run_retrieval
from adhoc.json_run_eval_helper import save_json_qres
from adhoc.retriever_if import RetrieverIF
from dataset_specific.beir_eval.beir_common import load_beir_queries_and_qrels
from taskman_client.task_proxy import get_task_manager_proxy
from trainer_v2.chair_logging import c_log


def run_retrieval_and_eval_on_beir(
        dataset: str, split: str, method: str,
        retriever: RetrieverIF,
        max_doc_per_list: int, do_not_report: bool = False):
    c_log.debug(f"Loading dataset")
    queries, qrels = load_beir_queries_and_qrels(dataset, split)
    c_log.info("%d queries", len(queries))
    c_log.info(f"run_retrieval")
    output = run_retrieval(retriever, queries, max_doc_per_list)
    run_name = f"{dataset}_{method}"
    save_json_qres(run_name, output)
    c_log.debug(f"run_evaluation")
    ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, output, [1, 10, 100, 1000])
    results2 = EvaluateRetrieval.evaluate_custom(qrels, output, [1, 10, 100, 1000], metric="r_cap")
    eval_res = {
        "NDCG@10": ndcg["NDCG@10"],
        "Recall@100": recall["Recall@100"],
        "R_cap@100": results2["R_cap@100"]
    }
    print(eval_res)
    if not do_not_report:
        metric = "NDCG@10"
        score = eval_res[metric]
        proxy = get_task_manager_proxy()
        proxy.report_number(method, score, dataset, metric)
        c_log.info(f"reported %s %f %s %s", method, score, dataset, metric)

    return eval_res
