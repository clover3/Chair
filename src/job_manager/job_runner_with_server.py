import os
import sys
import time

from data_generator.job_runner import WorkerInterface
from misc_lib import exist_or_mkdir
from taskman_client.task_proxy import get_task_manager_proxy, get_local_machine_name


class JobRunnerS:
    # worker_factory : gets output_path to as argument, returns object
    def __init__(self, working_path: str, max_job: int, job_name: str, worker_factory,
                 keep_on_exception=False,
                 max_job_per_worker=None,
                 worker_time_limit=None,
                 ):
        self.machine = get_local_machine_name()
        self.task_manager_proxy = get_task_manager_proxy()
        self.max_job = max_job
        self.working_path = working_path
        self.job_name = job_name
        self.out_path = os.path.join(working_path, job_name)
        self.worker_factory = worker_factory
        self.keep_on_exception = keep_on_exception
        self.max_job_per_worker = max_job_per_worker
        self.worker_time_limit = worker_time_limit
        self.st = time.time()
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
        n_job_done = 0
        while job_id is not None:
            try:
                worker.work(job_id)
            except Exception as e:
                update_type = "ERROR"
                msg = str(e)
                self.task_manager_proxy.sub_job_update(self.job_name, self.max_job, update_type, msg)
                if not self.keep_on_exception:
                    raise
            n_job_done += 1
            halt_run = self.check_halt_run(n_job_done)
            if halt_run:
                break
            job_id = self.report_done_and_pool_job(job_id)
            print("Job id : ", job_id)

    def check_halt_run(self, n_job_done):
        halt_run = self.max_job_per_worker is not None and n_job_done >= self.max_job_per_worker
        if not halt_run:
            if self.worker_time_limit is not None:
                elapsed = time.time() - self.st
                if elapsed > self.worker_time_limit:
                    halt_run = True

        return halt_run

    def run_one_job(self):
        worker = self.worker_factory(self.out_path)
        worker.work(int(sys.argv[1]))


class DummyWorker(WorkerInterface):
    def __init__(self, out_dir, work_fn):
        self.out_dir = out_dir
        self.work_fn = work_fn

    def work(self, job_id):
        return self.work_fn(job_id)


class JobRunnerF(JobRunnerS):
    # worker_factory : gets output_path to as argument, returns object
    def __init__(self, working_path: str, max_job: int, job_name: str, work_fn):
        def factor(out_dir):
            return DummyWorker(out_dir, work_fn)
        super(JobRunnerF, self).__init__(working_path, max_job, job_name, factor)
