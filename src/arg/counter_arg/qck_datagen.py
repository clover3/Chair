from arg.counter_arg.header import num_problems
from arg.qck.decl import QCKCandidate, QCKQuery
from arg.qck.instance_generator.qcknc_datagen import QCKInstanceGenerator
from arg.qck.qck_worker import QCKWorker
from data_generator.job_runner import JobRunner
from epath import job_man_dir


def start_job(job_name, split, candidate_dict, correct_d, qk_candidate):
    print("Loading data ....")

    def is_correct_fn(q: QCKQuery, c: QCKCandidate) -> bool:
        pair_id = q.query_id, c.id
        if pair_id in correct_d:
            return correct_d[pair_id]
        else:
            print("WARNING : key pair not found", pair_id)
            return False

    # transform payload to common QCK format
    generator = QCKInstanceGenerator(candidate_dict, is_correct_fn)

    print("Generate instances : ", split)

    def worker_factory(out_dir):
        return QCKWorker(qk_candidate,
                         generator,
                         out_dir)

    num_jobs = num_problems[split]
    runner = JobRunner(job_man_dir, num_jobs-1, job_name, worker_factory)
    runner.start()


