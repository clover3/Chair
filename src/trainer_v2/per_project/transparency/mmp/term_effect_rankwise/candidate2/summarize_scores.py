from cpath import output_path
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.fidelity_display_helper import collect_scores_and_save
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import term_align_candidate2_score_path, \
    mmp_root


def main():
    cand_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "candidate2.tsv")
    term_pair_list = [line.strip().split() for line in open(cand_path, "r")]
    save_path = term_align_candidate2_score_path()
    fidelity_save_dir = path_join(mmp_root(), "term_effect_space2", "fidelity")
    collect_scores_and_save(term_pair_list, fidelity_save_dir, save_path)


if __name__ == "__main__":
    main()
