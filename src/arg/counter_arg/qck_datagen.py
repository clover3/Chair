from typing import List, Dict

from arg.counter_arg.eval import prepare_eval_data, get_eval_payload_from_dp, retrieve_candidate, argu_is_correct
from arg.counter_arg.header import Passage, ArguDataID, num_problems
from arg.counter_arg.tf_datagen.qck_common import problem_to_qck_query, passage_to_candidate
from arg.qck.decl import QCKCandidate, QCKQuery
from arg.qck.qck_worker import QCKWorker
from arg.qck.qcknc_datagen import QCKInstanceGenerator
from data_generator.job_runner import JobRunner
from epath import job_man_dir
from list_lib import lmap


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


def load_base_resource(condition, split):
    problems, candidate_pool_d = prepare_eval_data(split)
    payload: List[Passage] = get_eval_payload_from_dp(problems)
    correct_d = {}
    candidate_dict: Dict[str, List[QCKCandidate]] = dict()
    for query, problem in zip(payload, problems):
        candidate_ids: List[ArguDataID] = retrieve_candidate(query, split, condition)
        candidate: List[Passage] = list([candidate_pool_d[x] for x in candidate_ids])
        qck_query = problem_to_qck_query(problem)
        qck_candidate_list = lmap(passage_to_candidate, candidate)
        candidate_dict[qck_query.query_id] = qck_candidate_list

        correct_list = list([argu_is_correct(problem, c) for c in candidate])
        for c, correct in zip(qck_candidate_list, correct_list):
            pair_id = qck_query.query_id, c.id
            correct_d[pair_id] = correct
    return candidate_dict, correct_d