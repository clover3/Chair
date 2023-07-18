from misc_lib import TimeProfiler
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.index_ranked_list2 import IRLProxy2
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import MMPGAlignPathHelper
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_mmp_split_w_deep_scores_train
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.te_measure_w_resource import \
    run_term_effect_over_term_pairs


def run_term_effect_w_path_helper(
        path_helper: MMPGAlignPathHelper,
        st, ed, disable_cache=False):
    partition_list = get_mmp_split_w_deep_scores_train()
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
        disable_cache
    )

    run_term_effect_over_term_pairs(
        irl_proxy, partition_list,
        per_candidate_conf.candidate_pair_path,
        per_corpus_conf.qtfs_path,
        per_candidate_conf.term_effect_save_dir,
        per_candidate_conf.fidelity_save_dir,
        time_profile,
        st, ed)
