from arg.perspectives.concat_rep_worker import PCConcatRepWorker
from data_generator.job_runner import JobRunner, sydney_working_dir

if __name__ == "__main__":
    def worker_factory(out_dir):
        rel_info_pickle_path = "pc_rel_dev_with_cpid"
        return PCConcatRepWorker(
            input_job_name="pc_rel_tfrecord_dev",
            pc_rel_info_pickle_name=rel_info_pickle_path,
            num_max_para=30,
            out_dir=out_dir
            )

    runner = JobRunner(sydney_working_dir, 665, "pc_concat_rep_dev", worker_factory)
    runner.start()


