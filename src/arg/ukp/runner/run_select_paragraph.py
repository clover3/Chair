from arg.ukp.select_paragraph import SelectParagraphWorker
from data_generator.job_runner import JobRunner, sydney_working_dir

if __name__ == "__main__":
    job_name = "ukp_paragraph_feature"
    runner = JobRunner(sydney_working_dir, 408, job_name, SelectParagraphWorker)
    runner.start()
