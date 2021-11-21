
import os

from cpath import output_path
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from tlm.data_gen.msmarco_doc_gen.gen_qtype.distil_preprocessing_common import ScoreTokenJoin

if __name__ == "__main__":
    split = "train"

    def worker_factory(out_dir):
        tfrecord_dir = os.path.join(job_man_dir, "MMD_10doc_train_first")
        prediction_dir = os.path.join(output_path, "MMD_10doc_train_first")
        return ScoreTokenJoin(tfrecord_dir, prediction_dir, out_dir)

    n_jobs = 367
    runner = JobRunnerS(job_man_dir, n_jobs, "MMD_2M_10doc_parse", worker_factory)
    runner.start()
