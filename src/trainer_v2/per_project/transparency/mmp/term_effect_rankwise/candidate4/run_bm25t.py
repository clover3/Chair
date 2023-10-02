from trainer_v2.per_project.transparency.mmp.bm25t_helper import load_mapping_from_align_scores, run_eval_with_bm25t
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand2_1_path_helper, \
    get_cand4_path_helper


def cand4():
    cut = 0.1
    mapping_val = 0.1

    ph = get_cand4_path_helper()
    table_path = ph.per_pair_candidates.fidelity_table_path
    mapping2 = load_mapping_from_align_scores(table_path, cut, mapping_val)

    table_name = f"cand4"
    run_name = f"bm25_{table_name}"
    dataset = "dev_sample1000"
    run_eval_with_bm25t(dataset, mapping2, run_name)


def main():
    cand4()


if __name__ == "__main__":
    main()
