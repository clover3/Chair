import json
import sys

from beir.retrieval.evaluation import EvaluateRetrieval

from dataset_specific.beir_eval.beir_common import load_beir_dataset
from dataset_specific.beir_eval.path_helper import get_beir_inv_index_path, get_beir_dl_path, \
    get_json_qres_save_path
from cpath import output_path
from misc_lib import path_join
from tab_print import tab_print
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.misc_common import save_tsv


def run_eval(dataset, method):
    metric = "NDCG@10"
    split = "test"
    c_log.info(f"Loading dataset")
    run_name = f"{dataset}_{method}"
    _, queries, qrels = load_beir_dataset(dataset, split)
    json_qres_save_path = get_json_qres_save_path(run_name)
    save_path = path_join(output_path, "per_line_eval", f"{run_name}.{metric}")
    f = open(save_path, "w")

    all_output = json.load(open(json_qres_save_path, "r"))

    table = []
    for query, inner in all_output.items():
        cur_res = {query: inner}
        ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, cur_res, [1, 10, 100, 1000])
        val = ndcg[metric]
        table.append((query, val))
    save_tsv(table, save_path)


def main():
    dataset = sys.argv[1]
    method = sys.argv[2]
    run_eval(dataset, method)


if __name__ == "__main__":
    main()
