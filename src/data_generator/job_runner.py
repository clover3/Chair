import logging
import sys
from abc import ABC, abstractmethod

from cache import *
from misc_lib import exist_or_mkdir
from sydney_manager import MarkedTaskManager
from tf_util.tf_logging import tf_logging


# JobRunner is responsible for recording which job is done and assigning jobs
# Worker is responsible for actually doing job


class WorkerInterface(ABC):
    @abstractmethod
    def work(self, job_id):
        pass


class JobRunner:
    # worker_factory : gets output_path to as argument, returns object
    def __init__(self, working_path, max_job, job_name, worker_factory):
        self.max_job = max_job
        self.working_path = working_path
        self.mark_path = os.path.join(working_path, job_name + "_mark")
        self.out_path = os.path.join(working_path, job_name)
        self.worker_factory = worker_factory
        exist_or_mkdir(self.mark_path)
        exist_or_mkdir(self.out_path)

    def start(self):
        if len(sys.argv) == 1:
            self.auto_runner()
        elif len(sys.argv) == 2:
            self.run_one_job()

    def auto_runner(self):
        mtm = MarkedTaskManager(self.max_job, self.mark_path, 1)
        worker = self.worker_factory(self.out_path)
        job_id = mtm.pool_job()
        print("Job id : ", job_id)
        while job_id is not None:
            worker.work(job_id)
            job_id = mtm.pool_job()
            print("Job id : ", job_id)

    def run_one_job(self):
        tf_logging.setLevel(logging.INFO)
        worker = self.worker_factory(self.out_path)
        worker.work(int(sys.argv[1]))

