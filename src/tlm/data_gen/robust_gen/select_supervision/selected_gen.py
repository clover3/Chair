from functools import partial

from cache import load_from_pickle
from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import AllSegmentAsDoc, RobustTrainGenSelected
from tlm.data_gen.robust_gen.robust_worker_w_data_id import RobustWorkerWDataID


def main():
    max_seq_length = 512
    score_d = load_from_pickle("robust_score_d2")
    encoder = AllSegmentAsDoc(max_seq_length)
    worker_factory = partial(RobustWorkerWDataID,
                             RobustTrainGenSelected(encoder, max_seq_length, score_d))
    runner = JobRunner(job_man_dir, 4, "robust_selected2", worker_factory)
    runner.start()


if __name__ == "__main__":
    main()
