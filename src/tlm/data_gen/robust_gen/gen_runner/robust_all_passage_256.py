from functools import partial

from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import AllSegmentAsDoc
from tlm.data_gen.robust_gen.gen_runner.robust_all_passage import RobustWorker
from tlm.data_gen.robust_gen.robust_generators import RobustPairwiseTrainGen


def generate_robust_all_seg_for_train():
    limited_length = 256
    encoder = AllSegmentAsDoc(limited_length)
    max_seq_length = 512
    worker_factory = partial(RobustWorker, RobustPairwiseTrainGen(encoder, max_seq_length))
    runner = JobRunner(job_man_dir, 4, "robust_all_passage_256", worker_factory)
    runner.start()


if __name__ == "__main__":
    generate_robust_all_seg_for_train()

