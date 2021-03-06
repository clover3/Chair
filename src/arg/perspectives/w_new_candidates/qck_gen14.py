import os
from typing import List, Dict

from arg.perspectives.load import load_claim_ids_for_split, d_n_claims_per_split2
from arg.perspectives.qck.qck_common import get_qck_candidate_from_ranked_list_path
from arg.perspectives.qck.qcknc_datagen import is_correct_factory
from arg.qck.decl import QKUnit
from arg.qck.instance_generator.qcknc_w_rel_score import QCKInstGenWScore
from arg.qck.qck_worker import QCKWorker
from cache import load_from_pickle
from cpath import output_path
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from list_lib import lmap
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def qck_gen(job_name, qk_candidate_name, candidate_ranked_list_path, kdp_ranked_list_path, split):
    claim_ids = load_claim_ids_for_split(split)
    cids: List[str] = lmap(str, claim_ids)
    qk_candidate: List[QKUnit] = load_from_pickle(qk_candidate_name)
    kdp_ranked_list: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(kdp_ranked_list_path)

    print("cids", len(cids))
    print("len(qk_candidate)", len(qk_candidate))
    print("Generate instances : ", split)
    generator = QCKInstGenWScore(get_qck_candidate_from_ranked_list_path(candidate_ranked_list_path),
                                 is_correct_factory(),
                                 kdp_ranked_list
                                 )
    qk_candidate_train: List[QKUnit] = list([qk for qk in qk_candidate if qk[0].query_id in cids])

    def worker_factory(out_dir):
        return QCKWorker(qk_candidate_train,
                         generator,
                         out_dir)

    num_jobs = d_n_claims_per_split2[split]
    runner = JobRunnerS(job_man_dir, num_jobs, job_name + "_" + split, worker_factory)
    runner.start()


def main():
    split = "train"
    job_name = "qck14"
    qk_candidate_name = "pc_qk2_filtered_" + split
    candidate_ranked_list_path = os.path.join(output_path,
                                    "perspective_experiments",
                                    "pc_qres", "{}.txt".format(split))

    kdp_ranked_list_path = os.path.join(output_path,
                                        "perspective_experiments",
                                        "clueweb_qres", "{}.txt".format(split))

    qck_gen(job_name, qk_candidate_name, candidate_ranked_list_path, kdp_ranked_list_path, split)


if __name__ == "__main__":
    main()
