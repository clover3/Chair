from functools import partial

from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import PassageSampling, RobustPointwiseTrainGenEx
from tlm.data_gen.run_robust_gen import RobustWorker


def main():
    max_seq_length = 512
    encoder = PassageSampling(max_seq_length)
    worker_factory = partial(RobustWorker, RobustPointwiseTrainGenEx(encoder, max_seq_length, "desc"))
    runner = JobRunner(job_man_dir, 4, "robust_train_desc", worker_factory)
    runner.start()


if __name__ == "__main__":
    main()
