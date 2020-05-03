from arg.perspectives.pc_rel.pc_rel_filter_worker import PCRelFilterWorker
from data_generator.job_runner import JobRunner, sydney_working_dir

if __name__ == "__main__":

    def worker_factory(out_dir):
        rel_info_pickle_path = "pc_rel_dev_with_cpid"
        return PCRelFilterWorker("pc_rel_tfrecord_dev", rel_info_pickle_path, out_dir)

    runner = JobRunner(sydney_working_dir, 696, "tf_rel_filter_B_dev", worker_factory)
    runner.start()


