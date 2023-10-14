from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.fidelity_display_helper import \
    collect_scores_and_save4
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand2_2_list_path_helper


def main():
    ph = get_cand2_2_list_path_helper()
    cand_path = ph.per_pair_candidates.candidate_pair_path
    term_pair_list = [line.strip().split() for line in open(cand_path, "r")]
    collect_scores_and_save4(
        term_pair_list,
        ph.per_pair_candidates.fidelity_save_dir,
        ph.per_pair_candidates.fidelity_table_path)


if __name__ == "__main__":
    main()