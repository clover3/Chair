import json
import sys
import time
from typing import List, Dict

from cache import save_list_to_jsonl
from cpath import output_path
from misc_lib import path_join
from taskman_client.job_group_proxy import JobGroupProxy
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bm25_runner.bm25_paramed import get_bm25_mmp_25_01_01
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.fidelity_helper import TermEffectPerQuery
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_te_save_path_base, load_qtf_index_train
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.term_effect_measure_mmp import ScoringModel, IRLProxy, \
    TermEffectMeasure


def get_te_candidat1_job_group_proxy():
    job_name = "te_cand1"
    max_job = 10000
    job_group = JobGroupProxy(job_name, max_job)
    return job_group


def term_effect_serial_for_list(sm, q_term, d_term, job_list):
    n_job = 0
    n_qd = 0
    n_query = 0
    st = time.time()
    for job_no in job_list:
        c_log.debug("Job %d", job_no)
        save_path = get_te_save_path_base(q_term, d_term, job_no)
        irl_proxy = IRLProxy(q_term)
        qtfs_index = load_qtf_index_train(job_no)
        tem = TermEffectMeasure(
            sm.get_updated_score_bm25,
            irl_proxy.get_irl,
            qtfs_index,
        )
        te_list: List[TermEffectPerQuery] = tem.term_effect_measure(q_term, d_term)

        out_itr = map(TermEffectPerQuery.to_json, te_list)
        save_list_to_jsonl(out_itr, save_path)

        n_qd += sum([len(te.changes) for te in te_list])
        n_query += len(te_list)
        n_job += 1
    ed = time.time()
    elapsed = ed - st
    time_per_q = elapsed / n_query if n_query else 0
    time_per_qd = elapsed / n_qd if n_qd else 0
    c_log.info(f"({q_term}, {d_term}) t={elapsed:.2f} t/q={time_per_q:.2f} t/qd={time_per_qd:.2f} n_jobs={n_job} n_query={n_query} n_qd={n_qd}")



def main():
    job_no = int(sys.argv[1])
    save_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "candidate1.tsv")
    todo_dict = get_missing_job_info()

    todo_list = [line.strip() for line in open(save_path, "r")]
    c_log.debug("load bm25")
    bm25 = get_bm25_mmp_25_01_01()
    c_log.debug("load bm25 Done")
    sm = ScoringModel(bm25.core.k1, bm25.core.b, bm25.core.avdl, bm25.term_idf_factor)

    job_group = get_te_candidat1_job_group_proxy()

    with job_group.sub_job_context(job_no):
        q_term, d_term = todo_list[job_no].split()
        c_log.info("Run %d-th line (%s, %s)", job_no, q_term, d_term)
        job_list = todo_dict[job_no]
        term_effect_serial_for_list(sm, q_term, d_term, job_list)


def get_missing_job_info():
    missing_list_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "candidate1_remain.jsonl")
    todo_dict: Dict[int, List[int]] = {}
    for line in open(missing_list_path, "r"):
        job_no_i, sub_job_todo = json.loads(line)
        todo_dict[job_no_i] = sub_job_todo
    return todo_dict


if __name__ == "__main__":
    main()