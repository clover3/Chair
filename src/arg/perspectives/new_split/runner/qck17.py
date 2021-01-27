from typing import List, Dict

from arg.perspectives.load import splits
from arg.perspectives.new_split.common import get_qids_for_split, split_name2, get_qck_candidate_for_split
from arg.perspectives.new_split.qk_common import load_all_qk
from arg.perspectives.qck.qcknc_datagen import is_correct_factory
from arg.qck.decl import QKUnit, QCKCandidate
from arg.qck.instance_generator.qcknc_datagen import QCKInstanceGenerator
from arg.qck.qck_worker import QCKWorker
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS


def qck_gen_w_ranked_list(job_name,
                          qk_candidates: List[QKUnit],
                          qck_candidates_dict: Dict[str, List[QCKCandidate]],
                          split):
    qids = list(get_qids_for_split(split_name2, split))
    print("Generate instances : ", split)
    generator = QCKInstanceGenerator(qck_candidates_dict, is_correct_factory())
    qk_candidates_for_split: List[QKUnit] = list([qk for qk in qk_candidates if qk[0].query_id in qids])
    print("{} of {} qk are used".format(len(qk_candidates_for_split), len(qk_candidates)))

    def worker_factory(out_dir):
        return QCKWorker(qk_candidates_for_split,
                         generator,
                         out_dir)

    num_jobs = len(qids)
    runner = JobRunnerS(job_man_dir, num_jobs, job_name + "_" + split, worker_factory)
    runner.start()


def main():
    job_name = "qck17"
    all_qk = load_all_qk()
    for split in splits:
        qck_candidates_dict: Dict[str, List[QCKCandidate]] = get_qck_candidate_for_split(split_name2, split)
        qck_gen_w_ranked_list(job_name,
                              all_qk,
                              qck_candidates_dict,
                              split)


if __name__ == "__main__":
    main()