from functools import partial

from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import PassageSampling
from tlm.data_gen.robust_gen.robust_generators import RobustPointwiseTrainGenEx
from tlm.data_gen.run_robust_gen import RobustWorker


def main():
    max_passage_length = 256
    encoder = PassageSampling(max_passage_length)
    max_seq_length = 512
    worker_factory = partial(RobustWorker, RobustPointwiseTrainGenEx(encoder, max_seq_length))
    runner = JobRunner(job_man_dir, 4, "robust_train_sampling_256", worker_factory)
    runner.start()
    ## td


if __name__ == "__main__":
    main()
