from arg.perspectives.PerspectiveParagraphTFRecordWorker import PerspectiveParagraphTFRecordWorker
from data_generator.job_runner import JobRunner, sydney_working_dir

if __name__ == "__main__":
    def worker_gen(out_dir):
        input_job_name = "perspective_paragraph_feature_dev"
        return PerspectiveParagraphTFRecordWorker(input_job_name, out_dir)

    runner = JobRunner(sydney_working_dir, 140, "pc_paragraph_tfrecord_dev", worker_gen)
    runner.start()