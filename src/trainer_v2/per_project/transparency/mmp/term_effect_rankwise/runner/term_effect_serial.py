import argparse
import sys
import time
from typing import List, Iterable, Callable, Dict, Tuple, Set
from cache import save_list_to_jsonl
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bm25_paramed import get_bm25_mmp_25_01_01
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_mmp_split_w_deep_scores
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.term_effect_measure import ScoringModel, TermEffectMeasure, \
    IRLProxy
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.fidelity_helper import TermEffectPerQuery
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import load_qtf_index, term_effect_dir, \
    get_te_save_path_base


parser = argparse.ArgumentParser()
parser.add_argument("--q_term", default="when")
parser.add_argument("--d_term", default="sunday")


def term_effect_serial_core(sm, q_term, d_term, irl_proxy, get_te_save_path):
    n_job = 0
    n_qd = 0
    n_query = 0
    st = time.time()
    for job_no in get_mmp_split_w_deep_scores():
        c_log.debug("Job %d", job_no)
        save_path = get_te_save_path(q_term, d_term, job_no)
        qtfs_index = load_qtf_index(job_no)
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
    args = parser.parse_args(sys.argv[1:])
    q_term = args.q_term
    d_term = args.d_term
    c_log.debug("load bm25")
    bm25 = get_bm25_mmp_25_01_01()
    c_log.debug("load bm25 Done")
    sm = ScoringModel(bm25.core.k1, bm25.core.b, bm25.core.avdl, bm25.term_idf_factor)
    irl_proxy = IRLProxy(q_term)
    term_effect_serial_core(sm, q_term, d_term, irl_proxy, get_te_save_path_base)


if __name__ == "__main__":
    main()
