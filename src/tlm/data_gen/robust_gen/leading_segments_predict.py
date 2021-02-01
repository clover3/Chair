from functools import partial

from arg.robust.qc_gen import RobustPredictGen, RobustWorker
from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import FirstAndRandom


def main():
    max_passage_length = 128
    num_segment = 4
    encoder = FirstAndRandom(max_passage_length, num_segment)
    max_seq_length = max_passage_length
    worker_factory = partial(RobustWorker, RobustPredictGen(encoder, max_seq_length))
    runner = JobRunner(job_man_dir, 4, "leading_segments_pred", worker_factory)
    runner.start()


if __name__ == "__main__":
    main()
