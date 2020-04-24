

from data_generator.job_runner import JobRunner, sydney_working_dir
from galagos.process_jsonl_doc_lines import JsonlWorker

num_dev_pc_jos = 11

if __name__ == "__main__":
    jsonl_path = "/mnt/nfs/work3/youngwookim/data/CLEF_eHealth_working/docs.jsonl"
    runner = JobRunner(sydney_working_dir, num_dev_pc_jos, "clef_0", lambda out_dir: JsonlWorker(jsonl_path, out_dir))
    runner.start()


