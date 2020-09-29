from data_generator.job_runner import JobRunner, sydney_working_dir
###
from galagos.process_jsonl_doc_lines import JsonlWorker
from misc_lib import ceil_divide

if __name__ == "__main__":
    num_lines = 231423
    block_size = 100
    num_jobs = ceil_divide(num_lines, block_size)
    jsonl_path = "/mnt/nfs/work3/youngwookim/data/counter_arg/q_res/ca_docs.jsonl"
    print("Start")
    runner = JobRunner(sydney_working_dir, num_jobs-1, "ca_docs", lambda out_dir: JsonlWorker(jsonl_path, out_dir))
    runner.start()


