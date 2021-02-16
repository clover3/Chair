from functools import partial

from arg.robust.qc_gen import RobustPredictGen, RobustWorker
from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import MultiWindow


def generate_robust_sero_for_train():
    total_sequence_length = 128 * 4
    src_window_size = 128
    encoder = MultiWindow(src_window_size, total_sequence_length)
    worker_factory = partial(RobustWorker, RobustPredictGen(encoder, total_sequence_length, 100, "desc"))
    runner = JobRunner(job_man_dir, 4, "RobustSero5_128_pred", worker_factory)
    runner.start()


if __name__ == "__main__":
    generate_robust_sero_for_train()

