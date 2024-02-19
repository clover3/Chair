from beir.retrieval.evaluation import EvaluateRetrieval

from dataset_specific.beir_eval.beir_common import beir_dataset_list_A, load_beir_queries_and_qrels
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.misc_common import save_tsv
from trec.ranked_list_util import ranked_list_to_dict
from trec.trec_parse import load_ranked_list_grouped
from cpath import output_path
from misc_lib import path_join


def run_save_per_query_eval(dataset, method):
    metric = "NDCG@10"
    split = "test"
    c_log.info(f"Loading dataset")
    queries, qrels = load_beir_queries_and_qrels(dataset, split)
    c_log.info(f"Done")
    rl_save_path = path_join(output_path, "ranked_list", f"{method}_{dataset}.txt")
    rlg = load_ranked_list_grouped(rl_save_path)
    all_output: dict = ranked_list_to_dict(rlg)

    run_name = f"{dataset}_{method}"
    save_path = path_join(output_path, "per_line_eval", f"{run_name}.{metric}")

    table = []
    for query, inner in all_output.items():
        cur_res = {query: inner}
        ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, cur_res, [1, 10, 100, 1000])
        val = ndcg[metric]
        table.append((query, val))
    save_tsv(table, save_path)


def main():

    c_log.info(__file__)
    method_list = [
        # "empty",
        "rr_mtc6_pep_tt17_10000",
        "rr_ce_msmarco_mini_lm"
    ]

    for method in method_list:
        for dataset in beir_dataset_list_A:
            run_save_per_query_eval(dataset, method)


if __name__ == "__main__":
    main()