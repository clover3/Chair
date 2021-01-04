

from typing import List

#
from arg.perspectives.load import load_train_claim_ids, get_claims_from_ids, load_dev_claim_ids, \
    load_claims_for_sub_split, d_n_claims_per_subsplit
from arg.perspectives.ppnc.resource import load_qk_candidate_train, load_qk_candidate_dev
from arg.qck.decl import QKUnit
from arg.qck.instance_generator.base import InstanceGenerator
from arg.qck.qck_worker import QCKWorker
from cache import load_from_pickle
from data_generator.job_runner import JobRunner
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from misc_lib import split_7_3


def start_generate_jobs_for_train_val(generator: InstanceGenerator,
                                      name_prefix):
    # claim ids split to train/val
    print("Loading data ....")
    d_ids: List[int] = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    train, val = split_7_3(claims)

    train_cids = {str(t['cId']) for t in train}
    val_cids = {str(t['cId']) for t in val}
    qk_candidate: List[QKUnit] = load_qk_candidate_train()
    print("Generate instances : train")
    qk_candidate_train: List[QKUnit] = list([qk for qk in qk_candidate if qk[0].query_id in train_cids])
    qk_candidate_val = list([qk for qk in qk_candidate if qk[0].query_id in val_cids])

    def worker_factory(out_dir):
        return QCKWorker(qk_candidate_train,
                         generator,
                         out_dir)

    runner = JobRunner(job_man_dir, 378, name_prefix + "_train", worker_factory)
    runner.start()

    print("Generate instances : val")

    def worker_factory(out_dir):
        return QCKWorker(qk_candidate_val,
                         generator,
                         out_dir)

    runner = JobRunner(job_man_dir, 162, name_prefix + "_val", worker_factory)
    runner.start()


def start_generate_jobs_for_val(generator: InstanceGenerator,
                                      name_prefix):
    # claim ids split to train/val
    print("Loading data ....")
    d_ids: List[int] = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    train, val = split_7_3(claims)

    val_cids = {str(t['cId']) for t in val}
    qk_candidate: List[QKUnit] = load_qk_candidate_train()
    qk_candidate_val = list([qk for qk in qk_candidate if qk[0].query_id in val_cids])

    print("Generate instances : val")

    def worker_factory(out_dir):
        return QCKWorker(qk_candidate_val,
                         generator,
                         out_dir)

    runner = JobRunner(job_man_dir, 162, name_prefix + "_val", worker_factory)
    runner.start()


def start_generate_jobs_for_train(generator: InstanceGenerator,
                                      name_prefix):
    # claim ids split to train/val
    print("Loading data ....")
    d_ids: List[int] = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    train, val = split_7_3(claims)

    train_cids = {str(t['cId']) for t in train}
    qk_candidate: List[QKUnit] = load_qk_candidate_train()
    qk_candidate_val = list([qk for qk in qk_candidate if qk[0].query_id in train_cids])

    print("Generate instances : train")

    def worker_factory(out_dir):
        return QCKWorker(qk_candidate_val,
                         generator,
                         out_dir)

    runner = JobRunner(job_man_dir, 378, name_prefix + "_train", worker_factory)
    runner.start()


def start_generate_jobs_for_dev(generator: InstanceGenerator,
                                      name_prefix):
    # claim ids split to train/val
    print("Loading data ....")
    d_ids: List[int] = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)

    cids = {str(t['cId']) for t in claims}
    qk_candidate: List[QKUnit] = load_qk_candidate_dev()
    qk_candidate_val = list([qk for qk in qk_candidate if qk[0].query_id in cids])

    print("Generate instances : dev")

    def worker_factory(out_dir):
        return QCKWorker(qk_candidate_val,
                         generator,
                         out_dir)

    runner = JobRunnerS(job_man_dir, 138, name_prefix + "_dev", worker_factory)
    runner.start()


def start_generate_jobs(generator: InstanceGenerator,
                        subsplit,
                        qk_candidate_name,
                        name_prefix):
    # claim ids split to train/val
    print("Loading data ....")
    claims = load_claims_for_sub_split(subsplit)

    valid_cids = {str(t['cId']) for t in claims}
    qk_candidate: List[QKUnit] = load_from_pickle(qk_candidate_name)
    qk_candidate_val = list([qk for qk in qk_candidate if qk[0].query_id in valid_cids])

    print("Generate instances :")
    print("split: ", subsplit)
    print("qk_candidate_name: ", qk_candidate_name)

    def worker_factory(out_dir):
        return QCKWorker(qk_candidate_val,
                         generator,
                         out_dir)

    num_job = d_n_claims_per_subsplit[subsplit]
    runner = JobRunner(job_man_dir, num_job, name_prefix + "_" + subsplit, worker_factory)
    runner.auto_runner()
