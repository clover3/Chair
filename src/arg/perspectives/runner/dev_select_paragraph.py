from arg.perspectives.select_paragraph_perspective import SelectParagraphWorker
from data_generator.job_runner import JobRunner, sydney_working_dir

if __name__ == "__main__":
    job_name = "perspective_paragraph_feature_dev"
    runner = JobRunner(sydney_working_dir,
                       140,
                       job_name,
                       SelectParagraphWorker.constructor_dev)
    runner.start()
