from typing import List

from arg.perspectives.load import load_claim_ids_for_split, d_n_claims_per_split2
from arg.perspectives.qck.qck_common import get_qck_candidate_from_ranked_list_path
from arg.perspectives.qck.qcknc_datagen import is_correct_factory
from arg.qck.decl import QKUnit
from arg.qck.instance_generator.qcknc_datagen import QCKInstanceGenerator
from arg.qck.qck_worker import QCKWorker
from cache import load_from_pickle
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from list_lib import lmap


def main():
    split = "train"
    job_name = "qck5"
    qk_candidate_name = "pc_qk2_train_filtered"
    ranked_list_path = "train"
    # Selected from doc_scorer_summarizer.py
    qck_gen_w_ranked_list(job_name, qk_candidate_name, ranked_list_path, split)


def qck_gen_w_ranked_list(job_name, qk_candidate_name, ranked_list_path, split):
    claim_ids = load_claim_ids_for_split(split)
    cids: List[str] = lmap(str, claim_ids)
    qk_candidate: List[QKUnit] = load_from_pickle(qk_candidate_name)
    print("Generate instances : ", split)
    generator = QCKInstanceGenerator(get_qck_candidate_from_ranked_list_path(ranked_list_path),
                                     is_correct_factory())
    qk_candidate_train: List[QKUnit] = list([qk for qk in qk_candidate if qk[0].query_id in cids])

    def worker_factory(out_dir):
        return QCKWorker(qk_candidate_train,
                         generator,
                         out_dir)

    num_jobs = d_n_claims_per_split2[split]
    runner = JobRunnerS(job_man_dir, num_jobs, job_name + "_" + split, worker_factory)
    runner.start()


if __name__ == "__main__":
    main()
