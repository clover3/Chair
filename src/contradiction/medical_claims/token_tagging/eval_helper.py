from contradiction.medical_claims.token_tagging.path_helper import get_sbl_qrel_path, get_save_path2
from trec.trec_eval_wrap_fn import trec_eval_wrap


def do_ecc_eval_w_trec_eval(run_name, label_type):
    prediction_path = get_save_path2(run_name, label_type)
    return trec_eval_wrap(get_sbl_qrel_path(), prediction_path, "map")


def main():
    do_ecc_eval_w_trec_eval("exact_match", "mismatch")


if __name__ == "__main__":
    main()