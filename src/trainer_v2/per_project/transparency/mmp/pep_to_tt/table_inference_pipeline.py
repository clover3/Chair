import logging
import os
import subprocess
import sys
import time

from omegaconf import OmegaConf

from misc_lib import path_join
from taskman_client.ssh_utils.squeue_dev import RemoteSSHClient
from taskman_client.ssh_utils.slurm_helper import submit_slurm_job
from taskman_client.task_proxy import get_task_manager_proxy
from trainer_v2.chair_logging import c_log


def check_files_existence(ssh_client, directory):
    # Get the list of files in the directory
    ls_data = ssh_client.ls(directory)
    if isinstance(ls_data, str):  # In case of error, ls_data would be an error message string
        print(f"Error listing directory: {ls_data}")
        return

    # Extract filenames
    filenames = {file['filename'] for file in ls_data}

    # Check for the existence of files 0.txt to 1000.txt
    all_files_exist = True
    missing_files = []

    for i in range(1001):
        filename = f"{i}.txt"
        if filename not in filenames:
            all_files_exist = False
            missing_files.append(filename)

    return all_files_exist, missing_files


def check_for_jobs_remote(hostname, username, job_name, interval=60, key_file=None, password=None):
    success = False
    while True:
        try:
            # Connect to the server
            with RemoteSSHClient(hostname, username, key_file, password) as ssh_client:

                # Check for jobs
                job_exists = False
                squeue_data = ssh_client.squeue(username)
                if isinstance(squeue_data, str):  # In case of error, squeue_data would be an error message string
                    logger.error(f"Error running squeue: {squeue_data}")
                else:
                    for job in squeue_data:
                        cur_job_name = job.get('NAME', '')
                        if job_name in cur_job_name:
                            job_exists = True
                            logger.info("Job %s exists", cur_job_name)
                            break

                # If no job remains, check for files
                if not job_exists:
                    success = True
                    break

        except Exception as e:
            logger.error(f"An error occurred: {e}")

        # Wait for the specified interval before the next check
        logger.debug("Sleep %s sec", interval)
        time.sleep(interval)
    return success


logger = logging.getLogger("SSH")
logger.setLevel(logging.DEBUG)
format_str = '%(levelname)s\t%(name)s \t%(asctime)s %(message)s'
formatter = logging.Formatter(format_str,
                              datefmt='%m-%d %H:%M:%S',
                              )
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.propagate = False


def build_combine_command(conf_path):
    src_path = f"src/trainer_v2/per_project/transparency/mmp/pep/term_pair_scoring_runner/combination_filtering_constant.py "
    combine_command = f"python {src_path} {conf_path}"
    return combine_command


def build_table_benchmark_command(table_path, run_name):
    src_path = "src/trainer_v2/per_project/transparency/mmp/retrieval_run/table_benchmark.py"
    combine_command = f"python {src_path} {table_path} {run_name}"
    return combine_command


unity_hostname = 'unity.rc.umass.edu'
unity_username = 'youngwookim_umass_edu'
sydney_base_path = "/mnt/nfs/work3/youngwookim/code/Chair"
unity_base_path = "/home/youngwookim_umass_edu/code/Chair/"


def table_eval_pipeline(conf_path):
    hostname = unity_hostname
    username = unity_username
    conf = OmegaConf.load(conf_path)
    combine_command = build_combine_command(conf_path)
    c_log.debug("combine_command: %s", combine_command)
    job_name = conf.short_job_name
    benchmark_cmd = build_table_benchmark_command(conf.table_save_path, job_name)
    c_log.debug("benchmark_cmd %s", benchmark_cmd)
    option_str = "--mem=48g "
    try:
        ret = check_for_jobs_remote(
            hostname, username, job_name, interval=600)
        if not ret:
            raise ValueError("check_for_jobs_and_files failed")
        ssh_client = RemoteSSHClient(hostname, username)
        ssh_client.submit_slurm_job(combine_command, job_name + "_combine")
        ssh_client.close()

        # run local script
        sync_table = "bash sync_table.sh"
        process = subprocess.run(sync_table, shell=True, capture_output=True, text=True)
        print(process.stdout, process.stderr, )
        print("return code", process.returncode)
        submit_slurm_job(None, benchmark_cmd, job_name + "_benchmark", sydney_base_path, option_str)
        proxy = get_task_manager_proxy()
        proxy.make_success_notification()
    except Exception as e:
        print("An error occurred:", e)
        proxy = get_task_manager_proxy()
        proxy.make_fail_notification()
        raise


def main():
    conf_path = "confs/experiment_confs/term_pair/mmp_freq100K_pair_pep_tt5.yaml"
    table_eval_pipeline(conf_path)


if __name__ == "__main__":
    main()