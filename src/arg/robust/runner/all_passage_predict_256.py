from functools import partial

from arg.robust.qc_gen import RobustWorker, RobustPredictGen
from data_generator.job_runner import JobRunner
from epath import job_man_dir
# There is four-level hierarchy for generating data for robust
# 1. JobRunner : this is basic job runner
# 2. Worker : RobustWorker  -> give range of queries to generator
# 3. Generator : RobustTrainGen, RobustPredictGen : Whether to make instance paired or not
# 4. Encoder : How the each query/document pair is encoded
from tlm.data_gen.adhoc_datagen import AllSegmentAsDoc


def generate_robust_all_seg_for_train():
    max_seq_length = 256
    encoder = AllSegmentAsDoc(max_seq_length)
    worker_factory = partial(RobustWorker, RobustPredictGen(encoder, max_seq_length))
    runner = JobRunner(job_man_dir, 4, "robust_all_passage_predict_256", worker_factory)
    runner.start()


if __name__ == "__main__":
    generate_robust_all_seg_for_train()

