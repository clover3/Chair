import argparse
import sys
from typing import List

from cache import save_list_to_jsonl
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bm25_paramed import get_bm25_mmp_25_01_01
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.term_effect_measure import ScoringModel, TermEffectMeasure, \
    IRLProxy
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.fidelity_helper import TermEffectPerQuery
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import load_qtf_index, term_effect_dir, \
    get_te_save_path
from misc_lib import path_join


def save_term_effect(te_list: List[TermEffectPerQuery], q_term, d_term, job_no):
    save_path = get_te_save_path(q_term, d_term, job_no)
    save_list_to_jsonl(te_list, save_path)


parser = argparse.ArgumentParser()
parser.add_argument("--q_term", default="when")
parser.add_argument("--d_term", default="sunday")
parser.add_argument("--job_no", default="1")


def main():
    args = parser.parse_args(sys.argv[1:])
    job_no = int(args.job_no)
    q_term = args.q_term
    d_term = args.d_term

    run_name = f"{q_term}_{d_term}_{job_no}"

    with JobContext(run_name):
        bm25 = get_bm25_mmp_25_01_01()
        sm = ScoringModel(bm25.core.k1, bm25.core.b, bm25.core.avdl, bm25.term_idf_factor)
        irl_proxy = IRLProxy(job_no)
        qtfs_index = load_qtf_index(job_no)
        tem = TermEffectMeasure(
            sm.get_updated_score_bm25,
            irl_proxy.get_irl,
            qtfs_index,
        )
        c_log.info("Done loading resources")
        c_log.info("Term effect for ({}, {}".format(q_term, d_term))
        te_list = tem.term_effect_measure(q_term, d_term)
        save_term_effect(te_list, q_term, d_term, job_no)
        c_log.info("Done evaluating effects")


if __name__ == "__main__":
    main()
