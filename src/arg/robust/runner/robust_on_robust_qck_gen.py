from typing import List

from arg.qck.decl import QKUnit
from arg.qck.instance_generator.qcknc_datagen import QCKInstanceGenerator
from arg.qck.qck_worker import QCKWorker
from arg.robust.qc_common import load_candidate_head_as_doc
from arg.robust.qck_gen import load_qk_robust_heldout, QRel
from data_generator.job_runner import JobRunner
from epath import job_man_dir


def main():
    qrel = QRel()
    qk_candidate: List[QKUnit] = load_qk_robust_heldout("651")
    candidate_dict = load_candidate_head_as_doc()
    generator = QCKInstanceGenerator(candidate_dict, qrel.is_correct)
    num_jobs = 250

    def worker_factory(out_dir):
        worker = QCKWorker(qk_candidate, generator, out_dir)
        return worker
    ##
    job_name = "robust_qck_2"
    runner = JobRunner(job_man_dir, num_jobs, job_name, worker_factory)
    runner.start()


if __name__ == "__main__":
    main()


