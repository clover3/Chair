import gzip
import json
import logging
import os.path

from krovetzstemmer import Stemmer

from cpath import output_path
from misc_lib import path_join, TimeProfiler
from taskman_client.job_group_proxy import JobGroupProxy
import sys
import time
from typing import List, Dict, Tuple

from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bm25_paramed import get_bm25_mmp_25_01_01
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_mmp_split_w_deep_scores
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.term_effect_measure import ScoringModel, \
    IRLProxy, IndexedRankedList
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.fidelity_helper import TermEffectPerQuery, \
    compare_fidelity, pearson_r_wrap
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import load_qtf_index, get_te_save_path2, \
    get_fidelity_save_path2


def get_te_candidate2_job_group_proxy():
    job_name = "te_cand2"
    max_job = 10000
    job_group = JobGroupProxy(job_name, max_job)
    return job_group


class CustomEncoder(json.JSONEncoder):
    def iterencode(self, o, _one_shot=False):
        if isinstance(o, float):
            yield format(o, '.4g')
        elif isinstance(o, list):
            yield '['
            first = True
            for value in o:
                if first:
                    first = False
                else:
                    yield ', '
                yield from self.iterencode(value)
            yield ']'
        else:
            yield from super().iterencode(o, _one_shot=_one_shot)


def save_list_to_jsonl(item_list, save_path):
    f_out = gzip.open(save_path, 'wt', encoding='utf8')
    for item in item_list:
        s = json.dumps(item, cls=CustomEncoder)
        f_out.write(s + "\n")
    f_out.close()


def compute_fidelity_change(te_list):
    fidelity_pair_list: List[Tuple[float, float]] = [compare_fidelity(te, pearson_r_wrap) for te in te_list]
    delta_sum = 0
    for t1, t2 in fidelity_pair_list:
        delta = t2 - t1
        delta_sum += delta
    return delta_sum



def write_f_change(q_term, d_term, score):
    f = open(get_fidelity_save_path2(q_term, d_term), "w")
    f.write(str(score))


def term_effect_serial_core(
        mmp_split_list: List[int],
        qtfs_index_per_job: Dict[int, Dict[str, List[str]]],
        sm,
        q_term: str, d_term: str,
        irl_proxy: IRLProxy,
        time_profile: TimeProfiler,
):
    n_job = 0
    n_qd = 0
    n_query = 0
    st = time.time()
    f_change_sum = 0
    for mmp_split_no in mmp_split_list:
        c_log.debug("MMP Split %d", mmp_split_no)
        save_path = get_te_save_path2(q_term, d_term, mmp_split_no)
        qtfs_index: Dict[str, List[str]] = qtfs_index_per_job[mmp_split_no]
        query_w_q_term: List[str] = qtfs_index[q_term]

        te_list: List[TermEffectPerQuery] = []
        for qid in query_w_q_term:
            time_profile.check("Load ranked list st")
            ranked_list: IndexedRankedList = irl_proxy.get_irl(qid)
            time_profile.check("Load ranked list ed")

            time_profile.check("Compute st")
            old_scores: List[float] = ranked_list.get_shallow_model_base_scores()
            entry_indices = ranked_list.get_entries_w_term(d_term)
            changes = []
            for entry_idx in entry_indices:
                entry = ranked_list.entries[entry_idx]
                new_score: float = sm.get_updated_score_bm25(q_term, d_term, entry)
                changes.append((entry_idx, new_score))

            target_scores = ranked_list.get_deep_model_scores()
            per_query = TermEffectPerQuery(target_scores, old_scores, changes)
            time_profile.check("Compute ed")
            te_list.append(per_query)

        time_profile.check("Save term effect st")
        out_itr = map(TermEffectPerQuery.to_json, te_list)
        save_list_to_jsonl(out_itr, save_path)
        time_profile.check("Save term effect ed")
        f_change = compute_fidelity_change(te_list)
        f_change_sum += f_change
        n_qd += sum([len(te.changes) for te in te_list])
        n_query += len(te_list)
        n_job += 1
    ed = time.time()
    elapsed = ed - st
    time_per_q = elapsed / n_query if n_query else 0
    time_per_qd = elapsed / n_qd if n_qd else 0
    c_log.info(f"({q_term}, {d_term}) t={elapsed:.2f} t/q={time_per_q:.2f} t/qd={time_per_qd:.2f} n_jobs={n_job} n_query={n_query} n_qd={n_qd}")
    return f_change_sum


def main():
    st = int(sys.argv[1])
    ed = int(sys.argv[2])
    job_name = f"run_candidate_{st}_{ed}"
    with JobContext(job_name):
        main_inner(ed, st)


def main_inner(ed, st):
    save_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "candidate2.tsv")
    c_log.setLevel(logging.DEBUG)
    todo_list = [line.strip() for line in open(save_path, "r")]
    c_log.debug("load bm25")
    bm25 = get_bm25_mmp_25_01_01()
    c_log.debug("load bm25 Done")
    sm = ScoringModel(bm25.core.k1, bm25.core.b, bm25.core.avdl, bm25.term_idf_factor)
    stemmer = Stemmer()
    mmp_split_list = get_mmp_split_w_deep_scores()
    c_log.info("{} Jobs".format(len(mmp_split_list)))
    c_log.info("Loading load_qtf_index")
    qtfs_index_per_job = {job_no: load_qtf_index(job_no) for job_no in mmp_split_list}
    c_log.info("Done")
    time_profile = TimeProfiler()
    irl_proxy = IRLProxy("unknown", time_profile)
    for i in range(st, ed):
        q_term, d_term = todo_list[i].split()
        if os.path.exists(get_fidelity_save_path2(q_term, d_term)):
            continue

        c_log.debug("Run %d-th line (%s, %s)", i, q_term, d_term)
        q_term_stm = stemmer.stem(q_term)
        d_term_stm = stemmer.stem(d_term)

        irl_proxy.set_new_q_term(q_term_stm)
        f_change = term_effect_serial_core(mmp_split_list, qtfs_index_per_job,
                                           sm, q_term_stm, d_term_stm, irl_proxy, time_profile)

        write_f_change(q_term, d_term, f_change)
        time_profile.print_time()
        time_profile.reset_time_acc()


if __name__ == "__main__":
    main()