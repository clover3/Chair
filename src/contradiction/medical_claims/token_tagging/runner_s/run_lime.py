import os
import sys
from typing import List

from contradiction.medical_claims.token_tagging.online_solver_common import make_ranked_list_w_solver2
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem, AlamriProblem
from contradiction.medical_claims.token_tagging.solvers.lime_solver import get_lime_solver_nli14_direct
from data_generator.job_runner import WorkerInterface
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from misc_lib import path_join, ceil_divide
from trainer_v2.chair_logging import c_log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class LimeWorker(WorkerInterface):
    def __init__(self, n_per_job, tag_type, target_idx, output_path):
        self.tag_type = tag_type
        self.output_path = output_path
        self.target_idx = target_idx
        self.problems: List[AlamriProblem] = load_alamri_problem()
        self.n_per_job = n_per_job
        self.solver = get_lime_solver_nli14_direct(target_idx)

    def work(self, job_id):
        st = job_id * self.n_per_job
        ed = st + self.n_per_job
        save_path = path_join(self.output_path, str(job_id))
        problems = self.problems[st:ed]
        make_ranked_list_w_solver2(problems, "lime", save_path, self.tag_type, self.solver)


def for_tag(tag_type, job_name):
    print(tag_type, job_name)
    target_idx = {"mismatch": 1,
                  "conflict": 2}[tag_type]
    problems: List[AlamriProblem] = load_alamri_problem()
    n_jobs = 20
    n_per_job = ceil_divide(len(problems), n_jobs)

    def factory(output_path):
        return LimeWorker(n_per_job, tag_type, target_idx, output_path)

    runner = JobRunnerS(job_man_dir, n_jobs, job_name, factory)
    runner.auto_runner()


def main2():
    c_log.info("Start {}".format(__file__))
    tag_type = "mismatch"
    job_name = "lime_token_tagging"
    for_tag(tag_type, job_name)


def main():
    c_log.info("Start {}".format(__file__))
    tag_type = "conflict"
    job_name = "lime_token_tagging_conflict_2"
    for_tag(tag_type, job_name)


if __name__ == "__main__":
    main()
