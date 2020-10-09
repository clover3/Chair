from typing import List

from arg.perspectives.load import load_claims_for_sub_split, d_n_claims_per_split
from arg.perspectives.qck.qcknc_datagen import get_eval_candidates_as_qck, is_correct_factory
from arg.qck.decl import QKUnit
from arg.qck.qck_worker import InstanceGenerator, QCKWorker
from arg.qck.qcknc_datagen import QCKInstanceGenerator
from cache import load_from_pickle
from data_generator.job_runner import JobRunner
from epath import job_man_dir


def run_jobs_with_qk_candidate(generator: InstanceGenerator,
                               sub_split,
                               qk_candidate_name,
                               name_prefix):

    n_claims_per_split = d_n_claims_per_split[sub_split]
    print("Loading data ....")

    claims = load_claims_for_sub_split(sub_split)

    cids = {str(t['cId']) for t in claims}
    qk_candidate: List[QKUnit] = load_from_pickle(qk_candidate_name)
    print("Generate instances : {}".format(sub_split))
    qk_candidate_val: List[QKUnit] = list([qk for qk in qk_candidate if qk[0].query_id in cids])

    def worker_factory(out_dir):
        return QCKWorker(qk_candidate_val,
                         generator,
                         out_dir)

    runner = JobRunner(job_man_dir, n_claims_per_split, name_prefix + "_" + sub_split, worker_factory)
    runner.start()


def main():
    generator = QCKInstanceGenerator(get_eval_candidates_as_qck("dev"), is_correct_factory())
    qk_candidate_name = "perspective_qk_stage2_dev"
    sub_split = "dev"
    run_jobs_with_qk_candidate(generator, sub_split, qk_candidate_name, "qcknc")


if __name__ == "__main__":
    main()