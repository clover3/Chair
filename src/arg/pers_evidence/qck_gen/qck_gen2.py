from arg.pers_evidence.runner.get_candidate_dict import load_bal_candidate
from arg.pers_evidence.runner.qc_gen import get_is_correct_fn
from arg.perspectives.load import d_n_pc_per_split
from arg.qck.qck_worker import QCKWorker
from arg.qck.qcknc_datagen import QCKInstanceGenerator
from cache import load_from_pickle
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerWServer
from misc_lib import tprint


def main():
    is_correct_fn = get_is_correct_fn()
    split = "train"
    split = "dev"
    qk_candidate = load_from_pickle("pc_evi_filtered_qk_{}".format(split))
    tprint("Loading candidates..")
    candidate_dict = load_bal_candidate(split)
    tprint("{} dict keys".format(len(candidate_dict)))

    tprint("Initializing generator..")
    generator = QCKInstanceGenerator(candidate_dict, is_correct_fn)
    num_jobs = d_n_pc_per_split[split]

    def worker_factory(out_dir):
        worker = QCKWorker(qk_candidate, generator, out_dir)
        return worker

    job_name = "pc_evi_qck2_{}".format(split)
    runner = JobRunnerWServer(job_man_dir, num_jobs, job_name, worker_factory)
    runner.start()


if __name__ == "__main__":
    main()
