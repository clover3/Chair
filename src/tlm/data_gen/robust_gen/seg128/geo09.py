from functools import partial

from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import GeoSampler
from tlm.data_gen.robust_gen.robust_generators import RobustPointwiseTrainGenEx
from tlm.data_gen.run_robust_gen import RobustWorker


def main():
    max_passage_length = 128
    g = 0.9
    encoder = GeoSampler(max_passage_length, g)
    max_seq_length = max_passage_length
    worker_factory = partial(RobustWorker, RobustPointwiseTrainGenEx(encoder, max_seq_length, "desc"))
    runner = JobRunner(job_man_dir, 4, "robust_geo09", worker_factory)
    runner.start()


if __name__ == "__main__":
    main()
