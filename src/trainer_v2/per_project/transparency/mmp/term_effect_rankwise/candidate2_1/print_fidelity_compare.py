from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.fidelity_display_helper import \
    collect_scores_and_save, collect_compare_scores
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand2_1_path_helper,get_cand2_1_spearman_path_helper
from cpath import output_path
from misc_lib import path_join


def main():
    ph = get_cand2_1_path_helper()
    ph2 = get_cand2_1_spearman_path_helper()
    cand_path = ph.per_pair_candidates.candidate_pair_path
    term_pair_list = [line.strip().split() for line in open(cand_path, "r")]
    compare_save_path = path_join(output_path, "msmarco", "passage", "compare.txt")
    collect_compare_scores(
        term_pair_list,
        ph.per_pair_candidates.fidelity_save_dir,
        ph2.per_pair_candidates.fidelity_save_dir,
        compare_save_path)


if __name__ == "__main__":
    main()
