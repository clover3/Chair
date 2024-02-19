from typing import List, Dict

from misc_lib import average
from tab_print import print_table
from table_lib import TablePrintHelper, DictCache
from trainer_v2.per_project.transparency.mmp.runner.measure_correlation import compute_ranked_list_correlation
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.fidelity_helper import pearson_r_wrap
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def load_ranked_list(name):
    print(f"loading from {name}")
    file_path = f"output/ranked_list/{name}.txt"
    l: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(file_path)
    return l


def main():
    correlation_fn = pearson_r_wrap

    def compute_avg_corr(l1, l2):
        corr_l = compute_ranked_list_correlation(l1, l2, correlation_fn)
        return average(corr_l)

    col_cache = DictCache(load_ranked_list)
    row_cache = DictCache(load_ranked_list)

    def get_corr_by_key(row_key, col_key):
        l1 = col_cache.get_val(col_key)
        l2 = row_cache.get_val(row_key)
        return compute_avg_corr(l1, l2)

    # Columns
    method_list = [
        "dev_sample1000_empty",
        "rr_mtc6_pep_tt17_10000_dev_sample1K_A",
    ]
    method_name_map = {
        "dev_sample1000_empty": "BM25",
        "rr_mtc6_pep_tt17_10000_dev_sample1K_A": "BM25T",
    }

    # Rows
    target_model_list = [
        "rr_ce_msmarco_mini_lm_dev1K_A",
        "rr_splade_dev1K_A",
        "rr_tas_b_dev1K_A",
        "rr_contriever_dev1K_A",
        "rr_contriever-msmarco_dev1K_A"
    ]
    target_model_name_map = {
        "rr_ce_msmarco_mini_lm_dev1K_A": "Cross Encoder",
        "rr_splade_dev1K_A": "Splade v2",
        "rr_tas_b_dev1K_A": "TAS-B",
        "rr_contriever_dev1K_A": "Contriever",
        "rr_contriever-msmarco_dev1K_A": "Contriever-MS MARCO"
    }
    printer = TablePrintHelper(
        method_list,
        target_model_list,
        method_name_map,
        target_model_name_map,
        get_corr_by_key,
        "Model",
    )

    print_table(printer.get_table())
    # head = ["Model"]
    # for m in method_list:
    #     head.append(method_name_map[m])
    #
    # table = [head]
    #
    # for target_model in target_model_list:
    #     target_rl = load_ranked_list(target_model)
    #     l1 = target_rl
    #     row = []
    #     row.append(target_model_name_map[target_model])
    #     for method_name in method_list:
    #         l2 = method_rl_d[method_name]
    #         corr_val = compute_avg_corr(l1, l2)
    #         row.append(corr_val)
    #     table.append(row)
    #
    # print_table(table)
    #


if __name__ == "__main__":
    main()
