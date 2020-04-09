from arg.ukp.select_paragraph import SelectParagraphWorker, Option, ANY_PARA_PER_DOC
from data_generator.job_runner import JobRunner, sydney_working_dir

if __name__ == "__main__":
    job_name = "ukp_paragraph_feature_2"
    option = Option(para_per_doc=ANY_PARA_PER_DOC)

    def worker_gen(out_dir):
        return SelectParagraphWorker(option, out_dir)
    runner = JobRunner(sydney_working_dir, 408, job_name, worker_gen)
    runner.start()
