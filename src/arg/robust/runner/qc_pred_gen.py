from functools import partial

from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import FirstSegmentAsDoc, RobustPredictGen
from tlm.data_gen.run_robust_gen import RobustWorker


def generate_robust_first_for_pred():
    doc_len = 256 + 3
    max_seq_length = 512
    encoder = FirstSegmentAsDoc(doc_len)
    worker_factory = partial(RobustWorker, RobustPredictGen(encoder, max_seq_length))
    runner = JobRunner(job_man_dir, 4, "robust_first_256_pred", worker_factory)
    runner.start()


if __name__ == "__main__":
    generate_robust_first_for_pred()

