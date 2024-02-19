from dataset_specific.beir_eval.beir_common import beir_dataset_name_map, beir_dataset_list_A
from misc_lib import average
from tab_print import print_table
from table_lib import TablePrintHelper
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

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

    target_to_be_explained = "rr_ce_msmarco_mini_lm"
    def get_corr_by_key(row_key_dataset, col_key_method):
        if col_key_method == "empty":
            run_name = f"{row_key_dataset}_{col_key_method}"
        else:
            run_name = f"{col_key_method}_{row_key_dataset}"

        l1 = load_ranked_list(run_name)
        l2 = load_ranked_list(f"{target_to_be_explained}_{row_key_dataset}")
        return compute_avg_corr(l1, l2)

    # Columns
    method_list = [
        "empty",
        "rr_mtc6_pep_tt17_10000",
    ]
    method_name_map = {
        "empty": "BM25",
        "rr_mtc6_pep_tt17_10000": "BM25T",
    }

    printer = TablePrintHelper(
        method_list,  # Column
        beir_dataset_list_A,
        method_name_map,
        beir_dataset_name_map,
        get_corr_by_key,
        "Model",
    )
    print_table(printer.get_table())

if __name__ == "__main__":
    main()
