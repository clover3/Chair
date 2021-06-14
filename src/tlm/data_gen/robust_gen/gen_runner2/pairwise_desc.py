from functools import partial

from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import LeadingN
from tlm.data_gen.robust_gen.robust_generators import RobustPairwiseTrainGen2
from tlm.data_gen.run_robust_gen import RobustWorker


def main():
    max_seq_length = 512
    encoder = LeadingN(max_seq_length, 1)
    worker_factory = partial(RobustWorker,
                             RobustPairwiseTrainGen2(encoder, max_seq_length, "desc"))
    runner = JobRunner(job_man_dir, 4, "robust_pairwise_head_desc", worker_factory)
    runner.start()


if __name__ == "__main__":
    main()

