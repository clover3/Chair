from functools import partial

from arg.robust.qc_gen import RobustPerQueryWorker_Legacy
from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.robust_gen.separate_encoder import RobustSeparateEncoder


def generate_robust_all_seg_for_predict():
    doc_max_length = 512
    worker_factory = partial(RobustPerQueryWorker_Legacy, RobustSeparateEncoder(doc_max_length, "desc", 1000, True))
    num_jobs = 250
    runner = JobRunner(job_man_dir, num_jobs-1, "robust_query_doc_pos_only", worker_factory)
    runner.start()


if __name__ == "__main__":
    generate_robust_all_seg_for_predict()
