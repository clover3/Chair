from contradiction.medical_claims.token_tagging.path_helper import get_sbl_qrel_path
from contradiction.medical_claims.token_tagging.print_score.stat_test import do_stat_test_map, do_stat_test_f1


def main():
    split = "test"
    judgment_path = get_sbl_qrel_path(split)
    metric = "map"
    tag = "mismatch"
    run1_list = ["nlits86", ]

    run2 = "nlits86_cip3"
    if metric == "map":
        do_stat_test_map(judgment_path, metric, run1_list, run2, tag)
    elif metric == "f1":
        do_stat_test_f1(metric, run1_list, run2, tag)


if __name__ == "__main__":
    main()