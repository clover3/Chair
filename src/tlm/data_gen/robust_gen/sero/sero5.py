from functools import partial

from data_generator.job_runner import JobRunner, sydney_working_dir
from tlm.data_gen.adhoc_datagen import MultiWindow, RobustPointwiseTrainGenEx
from tlm.data_gen.run_robust_gen import RobustWorker


def generate_robust_sero_for_train():
    total_sequence_length = 512 * 4
    src_window_size = 512
    encoder = MultiWindow(src_window_size, total_sequence_length)
    worker_factory = partial(RobustWorker, RobustPointwiseTrainGenEx(encoder, total_sequence_length, "desc"))
    runner = JobRunner(sydney_working_dir, 4, "RobustSero5", worker_factory)
    runner.start()


if __name__ == "__main__":
    generate_robust_sero_for_train()

