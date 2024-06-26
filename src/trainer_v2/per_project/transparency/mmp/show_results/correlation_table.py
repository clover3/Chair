from typing import List, Dict

from scipy.stats import kendalltau

from misc_lib import average
from tab_print import print_table
from table_lib import TablePrintHelper, DictCache
from trainer_v2.per_project.transparency.mmp.runner.measure_correlation import compute_ranked_list_correlation
from trainer_v2.per_project.transparency.mmp.show_results.other_fidelity import top_k_overlap, pairwise_preference_match
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.fidelity_helper import pearson_r_wrap, \
    kendalltau_r_wrap
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def load_ranked_list(name):
    print(f"loading from {name}")
    file_path = f"output/ranked_list/{name}.txt"
    l: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(file_path)
    return l


def get_correlation_fn(name):
    if name == "pearson":
        return pearson_r_wrap
    elif name == "kendalltau":
        return kendalltau_r_wrap
    elif name == "topk":
        return top_k_overlap
    elif name == "pairwise":
        return pairwise_preference_match


def main():
    col_cache = DictCache(load_ranked_list)
    row_cache = DictCache(load_ranked_list)

    for metric in [ "kendalltau", "pearson", "topk",  "pairwise",]:
        correlation_fn = get_correlation_fn(metric)
        print(metric)

        def compute_avg_corr(l1, l2):
            corr_l = compute_ranked_list_correlation(l1, l2, correlation_fn)
            return average(corr_l)

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



if __name__ == "__main__":
    main()
