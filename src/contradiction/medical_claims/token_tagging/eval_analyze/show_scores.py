import os

from contradiction.medical_claims.token_tagging.path_helper import get_sbl_qrel_path, get_save_path
from evals.metric_by_trec_eval import run_trec_eval


def get_map_for_run(run_name):
    qrel_path = get_sbl_qrel_path()
    trec_eval_path = os.environ["TREC_EVAL_PATH"]
    ranked_list_path = get_save_path(run_name)
    return run_trec_eval("map", qrel_path, ranked_list_path, trec_eval_path)


def main():
    method = "probe"
    tag = "mismatch"

    for sent_type in ["prem", "hypo"]:
        for label in ["entail", "neutral", "contradiction"]:
            try:
                run_name = f"{method}_{label}_{sent_type}_{tag}"
                print("{}\t{}".format(run_name, get_map_for_run(run_name)))
            except ValueError:
                pass


if __name__ == "__main__":
    main()