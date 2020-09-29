from typing import List

from arg.perspectives.load import load_train_claim_ids, get_claims_from_ids
from arg.qck.decl import QKUnit
from arg.qck.qck_worker import InstanceGenerator, QCKWorker
from cache import load_from_pickle
from data_generator.job_runner import JobRunner
from epath import job_man_dir
from misc_lib import split_7_3


def start_generate_jobs_for_train(generator: InstanceGenerator,
                                    qk_candidate_name,
                                      name_prefix):
    # claim ids split to train/val
    print("Loading data ....")
    d_ids: List[int] = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    train, val = split_7_3(claims)

    train_cids = {str(t['cId']) for t in train}
    qk_candidate: List[QKUnit] = load_from_pickle(qk_candidate_name)
    print("Generate instances : train")
    qk_candidate_train: List[QKUnit] = list([qk for qk in qk_candidate if qk[0].query_id in train_cids])

    def worker_factory(out_dir):
        return QCKWorker(qk_candidate_train,
                         generator,
                         out_dir)

    runner = JobRunner(job_man_dir, 378, name_prefix + "_train", worker_factory)
    runner.start()



def start_generate_jobs_for_val(generator: InstanceGenerator,
                                    qk_candidate_name,
                                      name_prefix):
    # claim ids split to train/val
    print("Loading data ....")
    d_ids: List[int] = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    _, val = split_7_3(claims)

    cids = {str(t['cId']) for t in val}
    qk_candidate: List[QKUnit] = load_from_pickle(qk_candidate_name)
    print("Generate instances : val")
    qk_candidate: List[QKUnit] = list([qk for qk in qk_candidate if qk[0].query_id in cids])

    def worker_factory(out_dir):
        return QCKWorker(qk_candidate,
                         generator,
                         out_dir)

    runner = JobRunner(job_man_dir, 162, name_prefix + "_val", worker_factory)
    runner.start()