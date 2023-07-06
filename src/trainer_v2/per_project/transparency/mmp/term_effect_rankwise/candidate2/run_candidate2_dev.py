import logging

from cpath import output_path
from misc_lib import path_join, TimeProfiler
import sys

from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_mmp_split_w_deep_scores_train, \
    get_mmp_split_w_deep_scores
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.te_measure_w_resource import \
    run_term_effect_over_term_pairs
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import mmp_root
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.term_effect_measure_mmp import IRLProxy


def main_inner(st, ed):
    c_log.setLevel(logging.INFO)
    split = "dev"
    partition_list = get_mmp_split_w_deep_scores(split)
    time_profile = TimeProfiler()
    irl_proxy = IRLProxy("unknown", time_profile)
    # tfs
    # shallow scores
    # deep scores
    fidelity_save_dir = path_join(mmp_root(), "term_effect_space2_dev", "fidelity")
    te_save_dir = path_join(mmp_root(), "term_effect_space2_dev", "content")

    qtfs_dir = path_join(mmp_root(), f"{split}_qtfs")

    term_pair_save_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "candidate2.tsv")

    run_term_effect_over_term_pairs(
        irl_proxy, partition_list, term_pair_save_path, qtfs_dir,
        te_save_dir, fidelity_save_dir,
        time_profile,
        st, ed)


def main():
    st = int(sys.argv[1])
    ed = int(sys.argv[2])
    job_name = f"run_candidate_{st}_{ed}"
    with JobContext(job_name):
        main_inner(st, ed)


if __name__ == "__main__":
    main()