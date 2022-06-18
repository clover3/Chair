from typing import List

from contradiction.medical_claims.token_tagging.eval_helper import do_ecc_eval_w_trec_eval
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from contradiction.medical_claims.token_tagging.problem_loader import get_valid_split_groups
from contradiction.medical_claims.token_tagging.tf_idf_discretize import idf_to_category_val
from list_lib import lmap, foreach
from trec.ranked_list_util import ensemble_ranked_list
from trec.trec_parse import load_ranked_list_grouped, RLG, write_rlg
from utils.ensemble import get_even_ensembler, EnsembleCoreIF, normalize, TransformEnsembleCore


def get_filter_dev_rlg_fn():
    valid_split_groups = get_valid_split_groups()
    def is_dev_query(qid):
        tokens = qid.split("_")
        return int(tokens[0]) in valid_split_groups

    def filter_rlg(rlg: RLG):
        out_rlg = {}
        for qid, items in rlg.items():
            if is_dev_query(qid):
                out_rlg[qid] = items
        return out_rlg
    return filter_rlg


def build_ensemble_ranked_list_and_save(run_name_list, combiner, tag_type, new_run_name):
    save_path = get_save_path2(new_run_name, tag_type)
    save_path_list = lmap(lambda name: get_save_path2(name, tag_type), run_name_list)
    rlg_list: List[RLG] = lmap(load_ranked_list_grouped, save_path_list)
    filter_rlg = get_filter_dev_rlg_fn()
    rlg_list = lmap(filter_rlg, rlg_list)
    rlg = ensemble_ranked_list(rlg_list, combiner, new_run_name)
    write_rlg(rlg, save_path)
    do_ecc_eval_w_trec_eval(new_run_name, tag_type)


def get_idf_and_sth_combiner(factor) -> EnsembleCoreIF:
        def v_grouping(v):
            v_abs = abs(v)
            sign = v / v_abs
            return sign * idf_to_category_val(v_abs)

        def idf_transform(scores):
            return lmap(v_grouping, scores)

        def normalize_fn(x):
            return normalize(x, factor)
        fn_list = [idf_transform, normalize_fn]
        return TransformEnsembleCore(fn_list)


def main2():
    run_name_list = [
        "tf_idf",
        "nlits40"
    ]
    output_run_name = "nlits40_ex"

    tag_type = "mismatch"
    ensemble_core = get_even_ensembler(len(run_name_list))
    build_ensemble_ranked_list_and_save(
        run_name_list,
        ensemble_core.combine,
        tag_type,
        output_run_name
    )


def main3():
    run_name_list = [
        "tf_idf",
        "nlits40"
    ]
    output_run_name = "nlits40_ex3_2"

    tag_type = "mismatch"
    ensemble_core = get_idf_and_sth_combiner()
    build_ensemble_ranked_list_and_save(
        run_name_list,
        ensemble_core.combine,
        tag_type,
        output_run_name
    )


def do_for_run(run_name):
    run_name_list = [
        "tf_idf",
        run_name
    ]
    factor = 1
    output_run_name = "{}_idf".format(run_name)
    print(output_run_name)
    tag_type = "mismatch"
    ensemble_core = get_idf_and_sth_combiner(factor)
    build_ensemble_ranked_list_and_save(
        run_name_list,
        ensemble_core.combine,
        tag_type,
        output_run_name
    )


def main():
    # todo = ["nlits40", "nlits45", "senli", "probe", "deletion", "pert_pred", "coattention"]
    todo = ["sbl"]
    foreach(do_for_run, todo)


if __name__ == "__main__":
    main()