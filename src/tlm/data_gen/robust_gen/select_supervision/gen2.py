from functools import partial

from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import AllSegmentAsDoc, RobustTrainGenSelected
from tlm.data_gen.robust_gen.robust_worker_w_data_id import RobustWorkerWDataID
from tlm.data_gen.robust_gen.select_supervision.common import load_score_set1


def main():
    max_seq_length = 512
    score_d = load_score_set1()
    encoder = AllSegmentAsDoc(max_seq_length)
    for target_selection in ["random_over_09", "best", "first_and_best", "best_or_over_09"]:
        worker_factory = partial(RobustWorkerWDataID,
                                 RobustTrainGenSelected(encoder, max_seq_length, score_d, "desc", target_selection))
        runner = JobRunner(job_man_dir, 3, "robust_selected2_{}".format(target_selection), worker_factory)
        runner.start()


if __name__ == "__main__":
    main()
