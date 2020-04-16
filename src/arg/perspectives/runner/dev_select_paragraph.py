from arg.perspectives.select_paragraph_perspective import SelectParagraphWorker
from data_generator.job_runner import JobRunner, sydney_working_dir
from galagos.query_runs_ids import Q_CONFIG_ID_BM25_10000

if __name__ == "__main__":
    job_name = "perspective_paragraph_feature_dev"


    def constructor(out_dir):
        return SelectParagraphWorker("dev", Q_CONFIG_ID_BM25_10000, out_dir)


    runner = JobRunner(sydney_working_dir, 140, job_name, constructor)
    runner.start()
