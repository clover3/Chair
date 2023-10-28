from beir.retrieval.evaluation import EvaluateRetrieval

from adhoc.json_run_eval_helper import save_json_qres
from adhoc.adhoc_retrieval import run_retrieval
from dataset_specific.beir_eval.beir_common import load_beir_dataset
from trainer_v2.chair_logging import c_log


def run_retrieval_and_eval(dataset, max_doc_per_list, method, qrels, queries, retriever):
    c_log.info(f"run_retrieval")
    output = run_retrieval(retriever, queries.items(), max_doc_per_list)
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


def run_retrieval_and_eval_on_beir(dataset, split, method, retriever, max_doc_per_list):
    c_log.info(f"Loading dataset")
    _, queries, qrels = load_beir_dataset(dataset, split)
    run_retrieval_and_eval(dataset, max_doc_per_list, method, qrels, queries, retriever)

