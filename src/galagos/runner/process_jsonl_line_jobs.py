from data_generator.job_runner import JobRunner
from epath import job_man_dir
from exec_lib import run_func_with_config
from galagos.process_jsonl_doc_lines import JsonlWorker2


def main(config):
    jsonl_path = config['jsonl_path']
    job_name = config['job_name']
    num_jobs = config['num_jobs']
    runner = JobRunner(job_man_dir, num_jobs, job_name, lambda out_dir: JsonlWorker2(jsonl_path, out_dir))
    runner.auto_runner()


if __name__ == "__main__":
    run_func_with_config(main)
