from functools import partial

from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import AllSegmentAsDoc, RobustPointwiseTrainGen
from tlm.data_gen.run_robust_gen import RobustWorker


def main():
    max_seq_length = 512
    encoder = AllSegmentAsDoc(max_seq_length)
    worker_factory = partial(RobustWorker, RobustPointwiseTrainGen(encoder, max_seq_length))
    runner = JobRunner(job_man_dir, 4, "robust_all_passage_pointwise", worker_factory)
    runner.start()


if __name__ == "__main__":
    main()
