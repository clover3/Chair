import random
from typing import List

from arg.pers_evidence.runner.get_candidate_dict import load_bal_candidate
from arg.pers_evidence.runner.qc_gen import get_is_correct_fn
from arg.perspectives.load import splits, d_n_pc_per_split
from arg.qck.decl import QKUnit, KDP
from arg.qck.qck_worker import QCKWorkerMultiple
from arg.qck.qcknc_datagen import QCKInstanceGenerator
from cache import load_from_pickle
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerWServer
from list_lib import left, right, lmap
from misc_lib import tprint, ceil_divide


def sample_kdps(qk_list: List[QKUnit]) -> List[QKUnit]:
    n = 4

    def sample(l: List[KDP]):
        random.shuffle(l)
        return l[:n]

    right_things = lmap(sample, right(qk_list))
    return list(zip(left(qk_list), right_things))


def main():
    is_correct_fn = get_is_correct_fn()
    for split in splits[:2]:
        qk_candidate = load_from_pickle("pc_evi_filtered_qk_{}".format(split))
        qk_candidate = sample_kdps(qk_candidate)
        tprint("Loading candidates..")
        candidate_dict = load_bal_candidate(split)
        tprint("{} dict keys".format(len(candidate_dict)))

        tprint("Initializing generator..")
        generator = QCKInstanceGenerator(candidate_dict, is_correct_fn)
        n_qk_per_job = 10
        num_jobs = ceil_divide(d_n_pc_per_split[split], n_qk_per_job)

        def worker_factory(out_dir):
            worker = QCKWorkerMultiple(qk_candidate, generator, n_qk_per_job, out_dir)
            return worker

        job_name = "pc_evi_qck3_{}".format(split)
        runner = JobRunnerWServer(job_man_dir, num_jobs, job_name, worker_factory)
        runner.start()


if __name__ == "__main__":
    main()
