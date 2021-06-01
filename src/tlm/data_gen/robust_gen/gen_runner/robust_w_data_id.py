from functools import partial

from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import AllSegmentAsDoc
from tlm.data_gen.robust_gen.robust_generators import RobustTrainGenWDataID
from tlm.data_gen.robust_gen.robust_worker_w_data_id import RobustWorkerWDataID


def main():
    max_seq_length = 512
    encoder = AllSegmentAsDoc(max_seq_length)
    worker_factory = partial(RobustWorkerWDataID, RobustTrainGenWDataID(encoder, max_seq_length))
    runner = JobRunner(job_man_dir, 4, "robust_w_data_id", worker_factory)
    runner.start()
    ## td


if __name__ == "__main__":
    main()
