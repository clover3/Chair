from adhoc.eval_helper.pytrec_helper import load_qrels_as_structure_from_any
from adhoc.resource.dataset_conf_helper import get_dataset_conf
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import RerankDatasetConf
from beir.retrieval.evaluation import EvaluateRetrieval

from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.misc_common import save_tsv
from trec.ranked_list_util import ranked_list_to_dict
from trec.trec_parse import load_ranked_list_grouped
from cpath import output_path
from misc_lib import path_join


def run_save_per_query_eval(dataset, method):
    if dataset in ["trec_dl19", "trec_dl20"]:
        metric = "NDCG@10"
        k = 10
    else:
        metric = "recip_rank"
        k = 1000
        if method == "empty":
            dataset = "dev1000"

    c_log.info(f"Loading dataset")
    dataset_conf: RerankDatasetConf = get_dataset_conf(dataset)
    dataset_name = dataset_conf.dataset_name
    judgment_path = dataset_conf.judgment_path
    qrels = load_qrels_as_structure_from_any(judgment_path)

    c_log.info(f"Done")
    rl_save_path = path_join(output_path, "ranked_list", f"{method}_{dataset_name}.txt")

    # ce_mini_lm_trec_dl19.txt
    # TREC_DL_2019_mct6_pep_tt17_10K.txt

    rlg = load_ranked_list_grouped(rl_save_path)
    all_output: dict = ranked_list_to_dict(rlg)

    run_name = f"{dataset_name}_{method}"
    save_path = path_join(output_path, "per_line_eval", f"{run_name}.{metric}")

    table = []
    for query, inner in all_output.items():
        cur_res = {query: inner}
        if metric == "recip_rank":
            ret = EvaluateRetrieval.evaluate_custom(qrels, cur_res, metric='mrr', k_values=[k])
            val = ret[f'MRR@{k}']
        else:
            ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, cur_res, [k])
            val = ndcg[metric]
        table.append((query, val))
    save_tsv(table, save_path)



def main():
    c_log.info(__file__)
    method_list = [
        "empty",
        "rr_mtc6_pep_tt17_10000",
        # "ce_msmarco_mini_lm"
    ]
    todo =  ["trec_dl19", "trec_dl20", "mmp_dev_sample1k_a"]
    # todo =  ["mmp_dev_sample1k_a"]
    for method in method_list:
        for dataset in todo:
            run_save_per_query_eval(dataset, method)


if __name__ == "__main__":
    main()