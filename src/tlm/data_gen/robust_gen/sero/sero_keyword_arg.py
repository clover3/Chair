import sys
from functools import partial

from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import MultiWindow, RobustPointwiseTrainGenEx
from tlm.data_gen.run_robust_gen import RobustWorker


def generate_robust_sero_for_train():
    window_size = int(sys.argv[1])
    n_window = int(sys.argv[2])
    total_sequence_length = window_size * n_window
    src_window_size = window_size
    encoder = MultiWindow(src_window_size, total_sequence_length)
    worker_factory = partial(RobustWorker, RobustPointwiseTrainGenEx(encoder, total_sequence_length, "title"))
    runner = JobRunner(job_man_dir, 4, "RobustSeroKeyword_{}_{}".format(window_size, n_window), worker_factory)
    runner.auto_runner()


if __name__ == "__main__":
    generate_robust_sero_for_train()

