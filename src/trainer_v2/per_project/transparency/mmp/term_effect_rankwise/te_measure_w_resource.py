import time
import os

from typing import List, Dict, Callable

from krovetzstemmer import Stemmer

from misc_lib import TimeProfiler, path_join
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.misc_common import save_number_to_file
from trainer_v2.per_project.transparency.mmp.bm25_paramed import get_bm25_mmp_25_01_01
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.fidelity_helper import compute_fidelity_change_pearson
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import load_q_term_index_from_qid_qtfs, \
    get_te_save_name, get_fidelity_save_name
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.te_measure_common import term_effect_per_partition, \
    compute_term_effect, save_te_list_to_gz_jsonl
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.term_effect_measure import IRLProxyIF, \
    ScoringModel
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.term_effect_measure_mmp import print_cur_memory


def term_effect_serial_core(
        partition_list: List[int],
        qtfs_index_per_job: Dict[int, Dict[str, List[str]]],
        sm,
        q_term: str, d_term: str,
        irl_proxy: IRLProxyIF,
        time_profile: TimeProfiler,
        get_te_save_path_fn: Callable[[str, str, int], str]
):
    n_job = 0
    n_qd = 0
    n_query = 0
    st = time.time()
    f_change_sum = 0
    for partition_no in partition_list:
        c_log.debug("MMP Split %d", partition_no)
        te_list = term_effect_per_partition(partition_no, qtfs_index_per_job, sm, q_term, d_term, irl_proxy,
                                            get_te_save_path_fn)
        f_change = compute_fidelity_change_pearson(te_list)
        f_change_sum += f_change
        n_qd += sum([len(te.changes) for te in te_list])
        n_query += len(te_list)
        n_job += 1
        print_cur_memory()
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
        return load_q_term_index_from_qid_qtfs(pickle_path)

    todo_list = [line.strip().split() for line in open(term_pair_save_path, "r")]

    def get_te_save_path(q_term, d_term, partition_no):
        save_name = get_te_save_name(q_term, d_term, partition_no)
        save_path = path_join(te_save_dir, save_name)
        return save_path

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
        save_name = get_fidelity_save_name(q_term, d_term)
        fidelity_save_path = path_join(fidelity_save_dir, save_name)
        if os.path.exists(fidelity_save_path):
            continue

        c_log.debug("Run %d-th line (%s, %s)", i, q_term, d_term)
        q_term_stm = stemmer.stem(q_term)
        d_term_stm = stemmer.stem(d_term)

        irl_proxy.set_new_q_term(q_term_stm)
        f_change = term_effect_serial_core(
            partition_list, qtfs_index_per_partition,
            sm, q_term_stm, d_term_stm, irl_proxy, time_profile,
            get_te_save_path
        )
        save_number_to_file(fidelity_save_path, f_change)
        # time_profile.print_time()
        time_profile.reset_time_acc()


def run_term_effect_over_term_pairs_per_partition(
        irl_proxy: IRLProxyIF, partition_no,
        term_pair_path,
        qtfs_dir,
        te_save_dir,
        fidelity_save_dir,
        time_profile,
        st, ed):
    def load_q_term_index(job_no):
        pickle_path = path_join(qtfs_dir, str(job_no))
        return load_q_term_index_from_qid_qtfs(pickle_path)

    todo_list = [line.strip().split() for line in open(term_pair_path, "r")]

    def get_te_save_path(q_term, d_term, partition_no):
        save_name = get_te_save_name(q_term, d_term, partition_no)
        save_path = path_join(te_save_dir, save_name)
        return save_path

    c_log.debug("load bm25")
    bm25 = get_bm25_mmp_25_01_01()
    c_log.debug("load bm25 Done")
    sm = ScoringModel(bm25.core.k1, bm25.core.b, bm25.core.avdl, bm25.term_idf_factor)
    stemmer = Stemmer()
    c_log.info("Loading load_q_term_index")
    q_term_index = {job_no: load_q_term_index(job_no) for job_no in [partition_no]}
    c_log.info("Done")
    for i in range(st, ed):
        q_term, d_term = todo_list[i]
        save_name = get_fidelity_save_name(q_term, d_term)
        fidelity_save_path = path_join(fidelity_save_dir, save_name)
        if os.path.exists(fidelity_save_path):
            continue

        c_log.debug("Run %d-th line (%s, %s)", i, q_term, d_term)
        q_term_stm = stemmer.stem(q_term)
        d_term_stm = stemmer.stem(d_term)

        irl_proxy.set_new_q_term(q_term_stm)
        save_path = get_te_save_path(q_term, d_term, partition_no)
        if os.path.exists(save_path):
            c_log.debug("%s exsits. skip computing", save_path)
            continue

        save_path = get_te_save_path(q_term, d_term, partition_no)
        te_list = compute_term_effect(irl_proxy, sm, q_term_index[partition_no], q_term_stm, d_term_stm)
        save_te_list_to_gz_jsonl(te_list, save_path)
        print_cur_memory()
