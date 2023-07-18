import sys

from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.fidelity_helper import TermEffectPerQuery, \
    pearson_r_wrap, compute_save_fidelity_from_te
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand2_1_path_helper
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_mmp_split_w_deep_scores_train


def main():
    st = int(sys.argv[1])
    ed = int(sys.argv[2])
    partition_list = get_mmp_split_w_deep_scores_train()
    path_helper = get_cand2_1_path_helper()
    per_candidate_conf = path_helper.per_pair_candidates
    te_save_dir = per_candidate_conf.term_effect_save_dir
    fidelity_save_dir = per_candidate_conf.fidelity_save_dir
    fidelity_fn = pearson_r_wrap

    todo_list = [line.strip().split() for line in open(per_candidate_conf.candidate_pair_path, "r")]

    def iterate_qd(todo_list, st, ed):
        for i in range(st, ed):
            yield todo_list[i]

    qd_itr = iterate_qd(todo_list, st, ed)
    compute_save_fidelity_from_te(fidelity_fn, fidelity_save_dir, partition_list, qd_itr, te_save_dir)


if __name__ == "__main__":
    main()