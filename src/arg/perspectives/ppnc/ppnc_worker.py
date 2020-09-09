import json
import os
from typing import List, Dict, Tuple, Callable

from arg.perspectives.load import load_train_claim_ids, get_claims_from_ids, load_dev_claim_ids
from arg.perspectives.ppnc.ppnc_decl import CPPNCGeneratorInterface
from cache import load_from_pickle
from data_generator.job_runner import JobRunner, WorkerInterface
from epath import job_man_dir
from misc_lib import split_7_3, DataIDManager, exist_or_mkdir


def start_generate_jobs_for_train_val(generator_functor: Callable[[Dict[int, List[Tuple[List[str], float]]]],
                                                                  CPPNCGeneratorInterface],
                                      writer,
                                      name_prefix):
    # claim ids split to train/val
    d_ids: List[int] = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    train, val = split_7_3(claims)
    data = load_from_pickle("pc_train_a_passages")
    entries, all_passages = data
    cid_to_passages: Dict[int, List[Tuple[List[str], float]]] = {claim['cId']: p for claim, p in entries}
    generator = generator_functor(cid_to_passages)

    print("Generate instances : train")

    def worker_factory(out_dir):
        return CPPNCWorker(train, generator, writer, out_dir)

    runner = JobRunner(job_man_dir, 378, name_prefix + "_train", worker_factory)
    runner.start()

    print("Generate instances : val")

    def worker_factory(out_dir):
        return CPPNCWorker(val, generator, writer, out_dir)

    runner = JobRunner(job_man_dir, 162, name_prefix + "_val", worker_factory)
    runner.start()


def start_generate_jobs_for_dev(generator_functor: Callable[[Dict[int, List[Tuple[List[str], float]]]],
                                CPPNCGeneratorInterface],
                                writer,
                                name_prefix):
    # claim ids split to train/val
    d_ids: List[int] = list(load_dev_claim_ids())
    claims = get_claims_from_ids(d_ids)
    data = load_from_pickle("pc_dev_a_passages")
    entries, all_passages = data
    cid_to_passages: Dict[int, List[Tuple[List[str], float]]] = {claim['cId']: p for claim, p in entries}
    generator = generator_functor(cid_to_passages)

    print("Generate instances : dev")

    def worker_factory(out_dir):
        return CPPNCWorker(claims, generator, writer, out_dir)

    runner = JobRunner(job_man_dir, 138, name_prefix + "_dev", worker_factory)
    runner.start()


class CPPNCWorker(WorkerInterface):
    def __init__(self,
                 claims,
                 generator: CPPNCGeneratorInterface,
                 writer,
                 out_dir):
        self.max_seq_length = 512
        self.generator = generator
        self.valid_jobs = list([c['cId'] for c in claims])
        print("Total of {} jobs".format(len(self.valid_jobs)))
        self.claims = claims
        self.out_dir = out_dir
        self.writer = writer

    def work(self, job_id):
        base = job_id * 10000
        data_id_manager = DataIDManager(base)
        insts: List = self.generator.generate_instances(self.claims[job_id], data_id_manager)
        print("{} instances".format(len(insts)))
        self.writer(insts, self.max_seq_length, os.path.join(self.out_dir, str(job_id)))

        info_dir = self.out_dir + "_info"
        exist_or_mkdir(info_dir)
        info_path = os.path.join(info_dir, str(job_id) + ".info")
        json.dump(data_id_manager.id_to_info, open(info_path, "w"))

