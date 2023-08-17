import logging
import sys

from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand2_1_path_helper
from misc_lib import TimeProfiler
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.index_ranked_list2 import IRLProxy2
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import MMPGAlignPathHelper
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_mmp_split_w_deep_scores_train
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.te_measure_w_resource import \
    run_term_effect_over_term_pairs, term_effect_per_partition
import time
import os

from typing import List, Dict, Callable

from krovetzstemmer import Stemmer

from misc_lib import TimeProfiler, path_join
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bm25_paramed import get_bm25_mmp_25_01_01
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.fidelity_helper import TermEffectPerQuery, \
    compute_fidelity_change_pearson
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import load_qtf_index_from_qid_qtfs
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.term_effect_measure import IRLProxyIF, \
    IndexedRankedList, ScoringModel
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.term_effect_measure_mmp import print_cur_memory



def term_effect_per_partition(
        partition_no, qtfs_index_per_job, sm,
        q_term, d_term,
        irl_proxy):
    qtfs_index: Dict[str, List[str]] = qtfs_index_per_job[partition_no]
    affected_qid_list: List[str] = qtfs_index[q_term]
    te_list: List[TermEffectPerQuery] = []
    for qid in affected_qid_list:
        ranked_list: IndexedRankedList = irl_proxy.get_irl(qid)
        old_scores: List[float] = ranked_list.get_shallow_model_base_scores()
        entry_indices = ranked_list.get_entries_w_term(d_term)
        changes = []
        for entry_idx in entry_indices:
            entry = ranked_list.entries[entry_idx]
            new_score: float = sm.get_updated_score_bm25(q_term, d_term, entry)
            changes.append((entry_idx, new_score))

        target_scores = ranked_list.get_deep_model_scores()
        per_query = TermEffectPerQuery(target_scores, old_scores, changes)
        te_list.append(per_query)
    out_itr = map(TermEffectPerQuery.to_json, te_list)
    return te_list


def term_effect_serial_core(
        partition_list: List[int],
        qtfs_index_per_job: Dict[int, Dict[str, List[str]]],
        sm,
        q_term: str, d_term: str,
        irl_proxy: IRLProxyIF,
        time_profile: TimeProfiler,
):
    n_job = 0
    n_qd = 0
    n_query = 0
    st = time.time()
    f_change_sum = 0
    for partition_no in partition_list:
        c_log.debug("MMP Split %d", partition_no)
        te_list = term_effect_per_partition(
            partition_no, qtfs_index_per_job, sm, q_term, d_term, irl_proxy,
                        )
        f_change = compute_fidelity_change_pearson(te_list)
        f_change_sum += f_change
        n_qd += sum([len(te.changes) for te in te_list])
        n_query += len(te_list)
        n_job += 1
        print_cur_memory()
        print(n_query, n_qd)

    ed = time.time()
    elapsed = ed - st
    time_per_q = elapsed / n_query if n_query else 0
    time_per_qd = elapsed / n_qd if n_qd else 0
    c_log.info(f"({q_term}, {d_term}) t={elapsed:.2f} t/q={time_per_q:.2f} "
               f"t/qd={time_per_qd:.2f} n_jobs={n_job} n_query={n_query} n_qd={n_qd}")
    print_cur_memory()
    return f_change_sum


def run_term_effect_over_term_pairs(
        irl_proxy: IRLProxyIF, partition_list,
        term_pair_save_path,
        qtfs_dir,
        te_save_dir,
        fidelity_save_dir,
        time_profile,
        st, ed):
    def load_qtf_index(job_no):
        pickle_path = path_join(qtfs_dir, str(job_no))
        return load_qtf_index_from_qid_qtfs(pickle_path)

    todo_list = [line.strip().split() for line in open(term_pair_save_path, "r")]

    c_log.debug("load bm25")
    bm25 = get_bm25_mmp_25_01_01()
    c_log.debug("load bm25 Done")
    sm = ScoringModel(bm25.core.k1, bm25.core.b, bm25.core.avdl, bm25.term_idf_factor)
    stemmer = Stemmer()
    c_log.info("{} partitions".format(len(partition_list)))
    c_log.info("Loading load_qtf_index")
    qtfs_index_per_partition = {job_no: load_qtf_index(job_no) for job_no in partition_list}
    c_log.info("Done")
    for i in range(st, ed):
        q_term, d_term = todo_list[i]
        c_log.debug("Run %d-th line (%s, %s)", i, q_term, d_term)
        q_term_stm = stemmer.stem(q_term)
        d_term_stm = stemmer.stem(d_term)

        irl_proxy.set_new_q_term(q_term_stm)
        f_change = term_effect_serial_core(
            partition_list, qtfs_index_per_partition,
            sm, q_term_stm, d_term_stm, irl_proxy, time_profile,
        )
        time_profile.reset_time_acc()


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


def main():
    path_helper = get_cand2_1_path_helper()
    c_log.setLevel(logging.DEBUG)
    st = int(sys.argv[1])
    ed = int(sys.argv[2])
    job_name = f"run_candidate_{st}_{ed}"
    with JobContext(job_name):
        run_term_effect_w_path_helper(path_helper, st, ed)


if __name__ == "__main__":
    main()
