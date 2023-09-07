from trainer_v2.per_project.transparency.mmp.runner.run_cand2_1_rerank import run_eval_with_bm25t, \
    load_mapping_from_align_scores
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand2_1_path_helper


def cand_2_1():
    ph = get_cand2_1_path_helper()
    table_path = ph.per_pair_candidates.fidelity_table_path
    dataset = "dev_sample1000"
    cut = 0.1
    mapping_val = 1
    table_name = f"cand2_1_cut01_mv1"
    run_name = f"bm25_{table_name}"
    mapping = load_mapping_from_align_scores(table_path, cut, mapping_val)
    run_eval_with_bm25t(dataset, mapping, run_name)


def main():
    cand_2_1()


if __name__ == "__main__":
    main()
