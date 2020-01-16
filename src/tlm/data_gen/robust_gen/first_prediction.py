from functools import partial

from data_generator.job_runner import JobRunner, sydney_working_dir
from tlm.data_gen.adhoc_datagen import RobustPredictGen, FirstSegmentAsDoc
from tlm.data_gen.run_robust_gen import RobustWorker


def generate_robust_first_for_prediction():
    max_seq_length = 512
    encoder = FirstSegmentAsDoc(max_seq_length)
    worker_factory = partial(RobustWorker, RobustPredictGen(encoder, max_seq_length))
    runner = JobRunner(sydney_working_dir, 4, "RobustFirstPred3", worker_factory)
    runner.start()


if __name__ == "__main__":
    generate_robust_first_for_prediction()

