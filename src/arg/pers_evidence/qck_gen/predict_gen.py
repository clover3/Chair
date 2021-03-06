from typing import List

from arg.pers_evidence.common import get_qk_per_split
from arg.pers_evidence.runner.get_candidate_dict import load_top_rank_candidate
from arg.pers_evidence.runner.qc_gen import get_is_correct_fn
from arg.perspectives.load import splits, d_n_pc_per_split
from arg.qck.decl import QKUnit
from arg.qck.instance_generator.qcknc_datagen import QCKInstanceGenerator
from arg.qck.qck_worker import QCKWorker
from epath import job_man_dir
from job_manager.job_runner2 import JobRunner2
from misc_lib import tprint


def main():
    is_correct_fn = get_is_correct_fn()
    qk_per_split = get_qk_per_split("pc_evidence_qk")
    for split in splits[1:]:
        qk_candidate: List[QKUnit] = qk_per_split[split]
        tprint("Loading candidates..")
        candidate_dict = load_top_rank_candidate(split)
        tprint("{} dict keys".format(len(candidate_dict)))

        tprint("Initializing generator..")
        generator = QCKInstanceGenerator(candidate_dict, is_correct_fn)
        num_jobs = d_n_pc_per_split[split]

        def worker_factory(out_dir):
            worker = QCKWorker(qk_candidate, generator, out_dir)
            return worker

        job_name = "pc_evi_qck_predict_{}".format(split)
        runner = JobRunner2(job_man_dir, num_jobs, job_name, worker_factory)
        runner.start()


if __name__ == "__main__":
    main()
