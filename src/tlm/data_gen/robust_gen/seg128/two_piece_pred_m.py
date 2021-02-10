from functools import partial

from arg.robust.qc_gen import RobustPerQueryWorker
from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.robust_gen.robust_train_gen_light import RobustPredictGenLight
from tlm.data_gen.robust_gen.seg_lib.segment_composer import ManyTwoPieceSegmentComposer


def generate_robust_all_seg_for_predict():
    max_seq_length = 128
    encoder = ManyTwoPieceSegmentComposer(max_seq_length)
    worker_factory = partial(RobustPerQueryWorker, RobustPredictGenLight(encoder, max_seq_length))
    num_jobs = 250
    runner = JobRunner(job_man_dir, num_jobs-1, "robust_two_piece_pred_m", worker_factory)
    runner.start()


if __name__ == "__main__":
    generate_robust_all_seg_for_predict()


##