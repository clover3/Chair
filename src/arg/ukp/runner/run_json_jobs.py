from data_generator.job_runner import JobRunner, sydney_working_dir
###
from galagos.process_jsonl_doc_lines import JsonlWorker


if __name__ == "__main__":
    num_dev_pc_jos = 128
    jsonl_path = "/mnt/nfs/work3/youngwookim/data/stance/ukp_para_query/docs_10.jsonl"
    runner = JobRunner(sydney_working_dir, num_dev_pc_jos, "ukp_10", lambda out_dir: JsonlWorker(jsonl_path, out_dir))
    runner.start()


