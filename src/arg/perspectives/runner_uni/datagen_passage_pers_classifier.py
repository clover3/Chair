import os
from typing import List, Dict, Tuple

from arg.perspectives.claim_lm.datagen_passage_pers_classifier import PairedInstance, generate_instances_functor, \
    write_records
from arg.perspectives.load import load_train_claim_ids, get_claims_from_ids
from cache import load_from_pickle
from data_generator.job_runner import WorkerInterface, JobRunner, sydney_working_dir
from misc_lib import split_7_3


def main():
    # claim ids split to train/val
    d_ids: List[int] = list(load_train_claim_ids())
    claims = get_claims_from_ids(d_ids)
    train, val = split_7_3(claims)

    def worker_factory(out_dir):
        return Worker(train, out_dir)

    runner = JobRunner(sydney_working_dir, 378, "passage_pers_classifier_train", worker_factory)
    runner.start()

    def worker_factory(out_dir):
        return Worker(val, out_dir)

    runner = JobRunner(sydney_working_dir, 162, "passage_pers_classifier_val", worker_factory)
    runner.start()


class Worker(WorkerInterface):
    def __init__(self, claims, out_dir):
        data = load_from_pickle("pc_train_a_passages")
        self.max_seq_length = 512
        entries, all_passages = data
        cid_to_passages: Dict[int, List[Tuple[List[str], float]]] = {claim['cId']: p for claim, p in entries}
        self.generate_instances = generate_instances_functor(cid_to_passages)
        self.valid_jobs = list([c['cId'] for c in claims])
        print("Total of {} jobs".format(len(self.valid_jobs)))
        self.claims = claims
        self.out_dir = out_dir

    def work(self, job_id):
        print("Generate instances : train")
        insts: List[PairedInstance] = self.generate_instances(self.claims[job_id])
        write_records(insts, self.max_seq_length, os.path.join(self.out_dir, str(job_id)))


if __name__ == "__main__":
    main()