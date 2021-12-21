import os

from cpath import output_path
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from tlm.qtype.gen_qtype.distil_preprocessing_common import ScoreTokenJoin2

if __name__ == "__main__":
    split = "train"

    def worker_factory(out_dir):
        tfrecord_dir = os.path.join(job_man_dir, "MMD_train_set_a")
        prediction_dir = os.path.join(output_path, "qtype", "MMD_Z_train_set_a")
        return ScoreTokenJoin2(tfrecord_dir, prediction_dir, out_dir)

    n_jobs = 37
    runner = JobRunnerS(job_man_dir, n_jobs, "MMD_Z_query_a_parse", worker_factory)
    runner.start()
