import logging
import sys

from misc_lib import TimeProfiler
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.index_ranked_list2 import IRLProxy2
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand2_1_path_helper
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_mmp_split_w_deep_scores_train
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.te_measure_w_resource import \
    run_term_effect_over_term_pairs_per_partition


def run_te(path_helper, st, ed, partition):
    time_profile = TimeProfiler()
    # tfs
    # shallow scores
    # deep scores
    per_corpus_conf = path_helper.per_corpus
    per_candidate_conf = path_helper.per_pair_candidates
    per_model_conf = path_helper.per_model
    irl_proxy = IRLProxy2(
        per_corpus_conf.shallow_scores_by_qid,
        per_model_conf.deep_scores_by_qid,
        per_corpus_conf.tfs_save_path_by_qid,
        time_profile,
    )

    run_term_effect_over_term_pairs_per_partition(
        irl_proxy, partition,
        per_candidate_conf.candidate_pair_path,
        per_corpus_conf.qtfs_path,
        per_candidate_conf.term_effect_save_dir,
        per_candidate_conf.fidelity_save_dir,
        time_profile,
        st, ed)


def main():
    path_helper = get_cand2_1_path_helper()
    c_log.setLevel(logging.DEBUG)
    st = int(sys.argv[1])
    ed = int(sys.argv[2])
    partition = int(sys.argv[3])
    partition_list = get_mmp_split_w_deep_scores_train()

    if partition not in partition_list:
        return

    job_name = f"pp_0_1000_{partition}"
    with JobContext(job_name):
        c_log.info("Partition %d", partition)
        run_te(path_helper, st, ed, partition)



if __name__ == "__main__":
    main()