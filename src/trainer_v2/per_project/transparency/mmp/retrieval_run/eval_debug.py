import json
import logging
import sys
from typing import Dict

from omegaconf import OmegaConf
from pytrec_eval import RelevanceEvaluator

from adhoc.eval_helper.pytrec_helper import load_qrels_as_structure_from_any
from dataset_specific.beir_eval.path_helper import get_json_qres_save_path
from misc_lib import average
from table_lib import tsv_iter
from trainer_v2.chair_logging import c_log


# This file should not contain dataset specific codes


def run_retrieval_eval_report_w_conf(conf):
    # Collect path and name from conf
    method = conf.method
    dataset_conf_path = conf.dataset_conf_path

    dataset_conf = OmegaConf.load(dataset_conf_path)
    dataset_name = dataset_conf.dataset_name
    queries_path = dataset_conf.queries_path
    metric = dataset_conf.metric
    judgment_path = dataset_conf.judgment_path

    queries = list(tsv_iter(queries_path))
    run_name = f"{dataset_name}_{method}"
    json_qres_save_path = get_json_qres_save_path(run_name)
    doc_score_d = json.load(open(json_qres_save_path, "r"))
    qrels: Dict[str, Dict[str, int]] = load_qrels_as_structure_from_any(judgment_path)

    evaluator = RelevanceEvaluator(qrels, {metric})
    score_per_query = evaluator.evaluate(doc_score_d)
    per_query_scores = [score_per_query[qid][metric] for qid in score_per_query]
    score = average(per_query_scores)
    print(f"metric:\t{score}")


def main():
    c_log.setLevel(logging.INFO)
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    run_retrieval_eval_report_w_conf(conf)


if __name__ == "__main__":
    main()
