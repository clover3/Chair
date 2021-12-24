import os

from data_generator.job_runner import WorkerInterface
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from tlm.qtype.gen_f_de.sampling import keep_all_label_ids_as_float, extract_sampler_w_float


class Worker(WorkerInterface):
    def __init__(self, input_dir, out_dir):
        self.out_dir = out_dir
        self.input_dir = input_dir

    def work(self, job_id):
        input_tfrecord_path = os.path.join(self.input_dir, str(job_id))
        save_path = os.path.join(self.out_dir, str(job_id))
        extract_sampler_w_float(input_tfrecord_path, save_path,
                                  keep_all_label_ids_as_float, 10)


if __name__ == "__main__":
    input_dir = os.path.join(job_man_dir, "MMD_train_f_de_distill2")

    def factory(out_dir):
        return Worker(input_dir, out_dir)

    n_jobs = 37
    runner = JobRunnerS(job_man_dir, n_jobs, "MMD_train_f_de_distill2_sampled", factory)
    runner.start()
