from arg.pf_common.ParagraphTFRecordWorker import ParagraphTFRecordWorker
from data_generator.job_runner import JobRunner, sydney_working_dir

if __name__ == "__main__":
    def worker_gen(out_dir):
        input_job_name = "perspective_paragraph_feature"
        return ParagraphTFRecordWorker(input_job_name, out_dir)

    runner = JobRunner(sydney_working_dir, 605, "pc_paragraph_tfrecord", worker_gen)
    runner.start()