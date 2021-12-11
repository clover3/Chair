
import os

from cpath import output_path
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from tlm.qtype.gen_qtype.distil_preprocessing_common import ScoreTokenJoin

if __name__ == "__main__":
    split = "train"

    def worker_factory(out_dir):
        tfrecord_dir = os.path.join(job_man_dir, "MMD_train_set_a")
        prediction_dir = os.path.join(output_path, "MMD_train_set_a")
        return ScoreTokenJoin(tfrecord_dir, prediction_dir, out_dir)

    n_jobs = 367
    runner = JobRunnerS(job_man_dir, n_jobs, "MMD_2M_query_a_parse", worker_factory)
    runner.start()
