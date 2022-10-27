import os

from contradiction.medical_claims.token_tagging.path_helper import get_sbl_vak_qrel_path, get_save_path, \
    get_sbl_qrel_path
from evals.metric_by_trec_eval import run_trec_eval
from tab_print import print_table


def get_map_for_run(run_name, split):
    qrel_path = get_sbl_qrel_path(split)
    trec_eval_path = os.environ["TREC_EVAL_PATH"]
    ranked_list_path = get_save_path(run_name)
    return run_trec_eval("map", qrel_path, ranked_list_path, trec_eval_path)


def show_for_probe():
    method = "probe"
    tag = "mismatch"

    for sent_type in ["prem", "hypo"]:
        for label in ["entail", "neutral", "contradiction"]:
            try:
                run_name = f"{method}_{label}_{sent_type}_{tag}"
                print("{}\t{}".format(run_name, get_map_for_run(run_name)))
            except ValueError:
                pass


def main():
    mismatch_run_list = ["random", "nlits86", "tf_idf", "psearch", "coattention", "senli", "deletion", "exact_match"]
    split = "test"
    table = []
    for run_name in mismatch_run_list:
        row = [run_name]
        for tag in ["mismatch", "conflict"]:
            try:
                file_name = f"{run_name}_{tag}"
                row.append(get_map_for_run(file_name, split))
            except ValueError:
                pass
            except FileNotFoundError as e:
                print(e)
                row.append("NA")
        table.append(row)
    print_table(table)


if __name__ == "__main__":
    main()