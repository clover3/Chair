from typing import List

from arg.qck.decl import QKUnit, QCKQuery, KDP
from arg.qck.instance_generator.qk_scoring_gen import QKInstanceGenerator
from arg.qck.qck_worker import QCKWorker
from cache import load_from_pickle
from data_generator.job_runner import JobRunner
from epath import job_man_dir


def main():
    qrel_path = "/home/youngwookim/Downloads/rob04-desc/qrels.rob04.txt"

    def is_correct(query: QCKQuery, kdp: KDP):
        return 0

    qk_candidate: List[QKUnit] = load_from_pickle("robust_on_clueweb_qk_candidate")
    generator = QKInstanceGenerator(is_correct)
    num_jobs = 250

    def worker_factory(out_dir):
        worker = QCKWorker(qk_candidate, generator, out_dir)
        return worker
    ##
    job_name = "robust_qk"
    runner = JobRunner(job_man_dir, num_jobs, job_name, worker_factory)
    runner.start()


if __name__ == "__main__":
    main()
