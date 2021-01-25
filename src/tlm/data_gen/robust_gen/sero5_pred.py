from functools import partial

from arg.robust.qc_gen import RobustPredictGen, RobustWorker
from data_generator.job_runner import JobRunner, sydney_working_dir
from tlm.data_gen.adhoc_datagen import MultiWindow


def generate_robust_sero_for_prediction():
    total_sequence_length = 512 * 4
    src_window_size = 512
    encoder = MultiWindow(src_window_size, total_sequence_length)
    worker_factory = partial(RobustWorker, RobustPredictGen(encoder, total_sequence_length, 100, "desc"))
    runner = JobRunner(sydney_working_dir, 4, "RobustSeroPred5", worker_factory)
    runner.start()


if __name__ == "__main__":
    generate_robust_sero_for_prediction()

