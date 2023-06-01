import sys
from cpath import output_path
from misc_lib import path_join
from taskman_client.job_group_proxy import JobGroupProxy
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bm25_paramed import get_bm25_mmp_25_01_01
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.runner.term_effect_serial import \
    term_effect_serial_core
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.term_effect_measure import ScoringModel


def get_te_candidat1_job_group_proxy():
    job_name = "te_cand1"
    max_job = 10000
    job_group = JobGroupProxy(job_name, max_job)
    return job_group


def main():
    job_no = int(sys.argv[1])
    n_per_job = 1
    save_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "candidate1.tsv")

    todo_list = [line.strip() for line in open(save_path, "r")]
    c_log.debug("load bm25")
    bm25 = get_bm25_mmp_25_01_01()
    c_log.debug("load bm25 Done")
    sm = ScoringModel(bm25.core.k1, bm25.core.b, bm25.core.avdl, bm25.term_idf_factor)

    job_group = get_te_candidat1_job_group_proxy()

    st = job_no * n_per_job
    ed = st + n_per_job
    for i in range(st, ed):
        with job_group.sub_job_context(i):
            q_term, d_term = todo_list[i].split()
            c_log.info("Run %d-th line (%s, %s)", i, q_term, d_term)
            term_effect_serial_core(sm, q_term, d_term)



if __name__ == "__main__":
    main()