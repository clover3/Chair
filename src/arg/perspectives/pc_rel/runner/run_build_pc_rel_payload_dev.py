from arg.perspectives.pc_rel.pc_rel_tfrecord_worker import PCRelTFRecordWorker
from data_generator.job_runner import JobRunner, sydney_working_dir

if __name__ == "__main__":
    def worker_gen(out_dir):
        input_job_name = "perspective_paragraph_feature_dev_11"
        return PCRelTFRecordWorker(input_job_name, out_dir)

    runner = JobRunner(sydney_working_dir, 695, "pc_rel_tfrecord_dev", worker_gen)
    runner.start()