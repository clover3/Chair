
from functools import partial

from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import MultiWindow
from tlm.data_gen.robust_gen.robust_generators import RobustPairwiseTrainGen2
from tlm.data_gen.run_robust_gen import RobustWorker


def main():
    total_sequence_length = 512 * 4
    src_window_size = 512
    encoder = MultiWindow(src_window_size, total_sequence_length)
    worker_factory = partial(RobustWorker,
                             RobustPairwiseTrainGen2(encoder, total_sequence_length, "desc", 1000, "pos_major_repeat_enum")
                             )
    runner = JobRunner(job_man_dir, 4, "leading4_512_desc_pairwise", worker_factory)
    runner.start()


if __name__ == "__main__":
    main()
