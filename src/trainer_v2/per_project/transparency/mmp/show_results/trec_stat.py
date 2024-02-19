from cpath import output_path
from dataset_specific.beir_eval.beir_common import beir_mb_dataset_list, beir_dataset_list_A
from misc_lib import path_join
from runnable.ttest_from_scores import ttest_from_scores_path
from tab_print import print_table


def main():
    metric = "NDCG@10"

    method1 = "empty"
    method2 = "rr_mtc6_pep_tt17_10000"

    table = []
    for dataset in ["trec_dl19", "trec_dl20", "mmp_dev_sample1K_A"]:
        if dataset in ["trec_dl19", "trec_dl20"]:
            metric = "NDCG@10"
        else:
            metric = "recip_rank"

        def get_path(method):
            run_name = f"{dataset}_{method}"
            return path_join(output_path, "per_line_eval", f"{run_name}.{metric}")

        try:
            diff, p_value = ttest_from_scores_path(get_path(method1), get_path(method2))
            out_row = (dataset, diff, p_value)
            table.append(out_row)
        except FileNotFoundError as e:
            print(e)

    print_table(table)


if __name__ == '__main__':
    main()
