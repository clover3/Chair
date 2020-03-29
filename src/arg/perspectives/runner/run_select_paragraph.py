from arg.perspectives.select_paragraph import SelectParagraphWorker
from data_generator.job_runner import JobRunner, sydney_working_dir

if __name__ == "__main__":
    job_name = "perspective_paragraph_feature"
    runner = JobRunner(sydney_working_dir, 1000, job_name, SelectParagraphWorker.constructor_train)
    runner.start()
