import os

from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from tf_util.tfrecord_convertor import extract_convertor_w_float
from tlm.qtype.gen_f_de.convertor import convert, Worker


def one_test_run():
    input_tfrecord_path = os.path.join(job_man_dir, "MMD_train_qe_de_distill_prob", str(2))
    save_tfrecord_path = os.path.join(job_man_dir, "MMD_train_f_de_distill_prob", str(2))
    extract_convertor_w_float(input_tfrecord_path, save_tfrecord_path, convert)


def dev_test_run():
    input_tfrecord_path = os.path.join(job_man_dir, "MMD_dev_qe_de", str(32))
    save_tfrecord_path = os.path.join(job_man_dir, "MMD_dev_f_de", str(32))
    extract_convertor_w_float(input_tfrecord_path, save_tfrecord_path, convert)


def run_jobs_for_train():
    n_jobs = 37
    split = "train"
    source_job = "MMD_train_qe_de_distill_prob"

    def factory(out_dir):
        return Worker(source_job, out_dir)

    runner = JobRunnerS(job_man_dir, n_jobs, "MMD_{}_f_de_distill_prob".format(split), factory)
    runner.start()


def run_jobs_for_train_base():
    n_jobs = 37
    split = "train"
    source_job = "MMD_train_qe_de_distill_base_prob"

    def factory(out_dir):
        return Worker(source_job, out_dir)

    runner = JobRunnerS(job_man_dir, n_jobs, "MMD_{}_f_de_distill_prob_base".format(split), factory)
    runner.start()


def run_jobs_for_dev():
    n_jobs = 50
    split = "dev"
    source_job = "MMD_{}_qe_de".format(split)

    def factory(out_dir):
        return Worker(source_job, out_dir)

    runner = JobRunnerS(job_man_dir, n_jobs, "MMD_{}_f_de".format(split), factory)
    runner.start()


def main():
    run_jobs_for_train_base()
    # run_jobs_for_train()
    # run_jobs_for_dev()


if __name__ == "__main__":
    main()