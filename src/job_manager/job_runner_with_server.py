import os
import sys

from misc_lib import exist_or_mkdir
from taskman_client.task_proxy import get_task_manager_proxy, get_local_machine_name


class JobRunnerS:
    # worker_factory : gets output_path to as argument, returns object
    def __init__(self, working_path: str, max_job: int, job_name: str, worker_factory):
        self.machine = get_local_machine_name()
        self.task_manager_proxy = get_task_manager_proxy()
        self.max_job = max_job
        self.working_path = working_path
        self.job_name = job_name
        self.out_path = os.path.join(working_path, job_name)
        self.worker_factory = worker_factory
        exist_or_mkdir(self.out_path)

    def start(self):
        if len(sys.argv) == 1:
            self.auto_runner()
        elif len(sys.argv) == 2:
            self.run_one_job()

    def pool_job(self) -> int:
        return self.task_manager_proxy.pool_job(self.job_name, self.max_job, self.machine)

    def report_done_and_pool_job(self, job_id) -> int:
        return self.task_manager_proxy.report_done_and_pool_job(self.job_name, self.max_job, self.machine, job_id)

    def auto_runner(self):
        worker = self.worker_factory(self.out_path)
        job_id = self.pool_job()
        print("Job id : ", job_id)
        while job_id is not None:
            worker.work(job_id)
            job_id = self.report_done_and_pool_job(job_id)
            print("Job id : ", job_id)

    def run_one_job(self):
        worker = self.worker_factory(self.out_path)
        worker.work(int(sys.argv[1]))
