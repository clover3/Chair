from arg.ukp.ukp_tfrecord_worker import UKPParagraphTFRecordWorker
from data_generator.job_runner import JobRunner, sydney_working_dir

if __name__ == "__main__":
    split = 'train'
    blind_topic = 'abortion'

    def worker_gen(out_dir):
        input_job_name = "ukp_paragraph_feature"
        return UKPParagraphTFRecordWorker(input_job_name, split, blind_topic, out_dir)

    runner = JobRunner(sydney_working_dir, 408, "ukp_paragraph_tfrecord_{}_{}".format(split, blind_topic), worker_gen)
    runner.start()
