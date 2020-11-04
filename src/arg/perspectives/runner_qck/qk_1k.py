from typing import Dict

from arg.perspectives.load import get_all_claim_d, d_n_claims_per_split2
from arg.perspectives.runner_qck.qkgen_from_db import QKGenFromDB
from data_generator.job_runner import JobRunner
from epath import job_man_dir
from exec_lib import run_func_with_config


def main(config):
    q_res_path = config['q_res_path']
    split = config['split']
    query_d: Dict[int, str] = get_all_claim_d()

    def worker_gen(out_dir):
        qkgen = QKGenFromDB(q_res_path, query_d, out_dir)
        return qkgen

    num_jobs = d_n_claims_per_split2[split]
    runner = JobRunner(job_man_dir, num_jobs, config['job_name'], worker_gen)
    runner.auto_runner()


# print

if __name__ == "__main__":
    run_func_with_config(main)
