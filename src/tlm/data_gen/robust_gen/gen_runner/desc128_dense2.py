from functools import partial

from arg.robust.qc_gen import RobustWorker, RobustPredictGenPrecise
from data_generator.job_runner import JobRunner
from epath import job_man_dir
# There is four-level hierarchy for generating data for robust
# 1. JobRunner : this is basic job runner
# 2. Worker : RobustWorker  -> give range of queries to generator
# 3. Generator : RobustTrainGen, RobustPredictGen : Whether to make instance paired or not
# 4. Encoder : How the each query/document pair is encoded
from tlm.data_gen.adhoc_datagen import OverlappingSegmentsEx


def generate_robust_all_seg_for_predict():
    max_seq_length = 128
    step_size = 16
    encoder = OverlappingSegmentsEx(max_seq_length, step_size)
    worker_factory = partial(RobustWorker, RobustPredictGenPrecise(encoder, max_seq_length, 100, "desc"))

    runner = JobRunner(job_man_dir, 4, "robust_predict_desc_128_step16_2", worker_factory)
    runner.start()


if __name__ == "__main__":
    generate_robust_all_seg_for_predict()
