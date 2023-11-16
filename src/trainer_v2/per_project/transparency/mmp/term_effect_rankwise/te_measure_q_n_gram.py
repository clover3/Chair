import os
import pickle
import time
from collections import Counter, defaultdict

from krovetzstemmer import Stemmer

from table_lib import tsv_iter
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.misc_common import save_number_to_file
from trainer_v2.per_project.transparency.mmp.bm25_runner.bm25_paramed import get_bm25_mmp_25_01_01
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.fidelity_helper import \
    compute_fidelity_change_pearson
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.te_measure_common import compute_term_effect, save_te_list_to_gz_jsonl
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.term_effect_measure import IRLProxyIF, \
    ScoringModelNGram
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.term_effect_measure_mmp import print_cur_memory
from typing import List, Dict, Tuple
from misc_lib import TimeProfiler
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.index_ranked_list2 import IRLProxy2
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import MMPGAlignPathHelper
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_mmp_split_w_deep_scores_train


def term_effect_iter_partitions(
        partition_list: List[int],
        per_partition_work,
        job_desc: str,
):
    stat = Counter()
    st = time.time()
    f_change_sum = 0
    for partition_no in partition_list:
        c_log.debug("MMP Split %d", partition_no)
        te_list = per_partition_work(partition_no)
        f_change = compute_fidelity_change_pearson(te_list)
        f_change_sum += f_change
        stat["n_query"] += len(te_list)
        stat["n_qd"] += sum([len(te.changes) for te in te_list])
        stat["n_job"] += 1
        print_cur_memory()
    ed = time.time()
    elapsed = ed - st
    print_stat(stat, elapsed, job_desc)
    return f_change_sum


def print_stat(stat, elapsed, job_desc):
    n_query = stat["n_query"]
    n_job = stat["n_job"]
    n_qd = stat["n_qd"]
    time_per_q = elapsed / n_query if n_query else 0
    time_per_qd = elapsed / n_qd if n_qd else 0
    c_log.info(f"{job_desc} t={elapsed:.2f} t/q={time_per_q:.2f} "
               f"t/qd={time_per_qd:.2f} n_jobs={n_job} n_query={n_query} n_qd={n_qd}")
    print_cur_memory()


class NGramNormalizer:
    def __init__(self):
        self.stemmer = Stemmer()

    def __call__(self, term):
        tokens = term.lower().split()
        tokens = [self.stemmer.stem(t) for t in tokens]
        return " ".join(tokens)


def run_term_effect_over_term_pairs(
        irl_proxy: IRLProxyIF, partition_list,
        term_pair_save_path,
        qterm_index_dir,
        te_save_dir,
        fidelity_save_dir,
        time_profile,
        st, ed):

    def load_q_term_index(job_no):
        pickle_f = open(path_join(qterm_index_dir, str(job_no)), "rb")
        d = pickle.load(pickle_f)
        dd = defaultdict(list)
        dd.update(d)
        return dd

    todo_list: List[Tuple] = list(tsv_iter(term_pair_save_path))

    def get_te_save_path(term_pair_idx, partition_no):
        save_name = f"{term_pair_idx}_{partition_no}.jsonl.gz"
        save_path = path_join(te_save_dir, save_name)
        return save_path

    c_log.debug("load bm25")
    bm25 = get_bm25_mmp_25_01_01()
    c_log.debug("load bm25 Done")
    sm = ScoringModelNGram(bm25.core.k1, bm25.core.b, bm25.core.avdl, bm25.term_idf_factor)
    normalizer = NGramNormalizer()
    c_log.info("{} partitions".format(len(partition_list)))
    c_log.info("Loading load_qtf_index")
    q_term_index_index_per_partition: Dict[int, Dict[str, List]]\
        = {job_no: load_q_term_index(job_no) for job_no in partition_list}
    c_log.info("Done")
    for i in range(st, ed):
        q_term, d_term = todo_list[i]
        save_name = str(i)
        fidelity_save_path = path_join(fidelity_save_dir, save_name)
        if os.path.exists(fidelity_save_path):
            continue

        c_log.debug("Run %d-th line (%s, %s)", i, q_term, d_term)
        q_term_norm = normalizer(q_term)
        d_term_norm = normalizer(d_term)

        def work_per_partition(partition_no):
            q_term_index = q_term_index_index_per_partition[partition_no]
            save_path = get_te_save_path(i, partition_no)
            te_list = compute_term_effect(
                irl_proxy, sm, q_term_index, q_term_norm, d_term_norm)
            save_te_list_to_gz_jsonl(te_list, save_path)
            return te_list

        irl_proxy.set_new_q_term(q_term_norm)
        job_desc = f"{i} ({q_term}, {d_term}) "
        f_change = term_effect_iter_partitions(
            partition_list, work_per_partition, job_desc,
        )
        save_number_to_file(fidelity_save_path, f_change)


def run_te_config_wrap(
        path_helper: MMPGAlignPathHelper,
        st, ed,
        disable_cache=False):
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
        per_candidate_conf.q_term_index_path,
        per_candidate_conf.term_effect_save_dir,
        per_candidate_conf.fidelity_save_dir,
        time_profile,
        st, ed)


