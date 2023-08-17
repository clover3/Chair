import sys

from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.fidelity_helper import TermEffectPerQuery, \
    spearman_r_wrap, compute_save_fidelity_from_te
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand2_1_spearman_path_helper, \
    get_mmp_galign_path_helper
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_mmp_split_w_deep_scores_train, \
    get_mmp_split_w_deep_scores
from cpath import output_path, yconfig_dir_path
from misc_lib import path_join


def compute_spearman_fidelity(ed, per_candidate_config_path, st):
    partition_list = get_mmp_split_w_deep_scores("train")
    per_corpus_config_path = path_join(yconfig_dir_path, "mmp_train.yaml")
    per_model_config_path = path_join(yconfig_dir_path, "mmp1.yaml")
    path_helper = get_mmp_galign_path_helper(
        per_corpus_config_path,
        per_model_config_path,
        per_candidate_config_path,
    )
    per_candidate_conf = path_helper.per_pair_candidates
    te_save_dir = per_candidate_conf.term_effect_save_dir
    fidelity_save_dir = per_candidate_conf.fidelity_save_dir
    fidelity_fn = spearman_r_wrap
    todo_list = [line.strip().split() for line in open(per_candidate_conf.candidate_pair_path, "r")]

    def iterate_qd(todo_list, st, ed):
        for i in range(st, ed):
            yield todo_list[i]

    qd_itr = iterate_qd(todo_list, st, ed)
    compute_save_fidelity_from_te(fidelity_fn, fidelity_save_dir, partition_list, qd_itr, te_save_dir)


def main():
    st = int(sys.argv[1])
    ed = int(sys.argv[2])
    per_candidate_config_path = path_join(yconfig_dir_path, "candidate2_spearman.yaml")

    compute_spearman_fidelity(ed, per_candidate_config_path, st)


if __name__ == "__main__":
    main()
