import sys
from abc import ABC, abstractmethod

from base_type import FilePath
from cache import *
from misc_lib import exist_or_mkdir
from sydney_manager import MarkedTaskManager

# JobRunner is responsible for recording which job is done and assigning jobs
# Worker is responsible for actually doing job

sydney_working_dir: FilePath = FilePath("/mnt/nfs/work3/youngwookim/data/bert_tf")


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
        self.done_path = os.path.join(working_path, job_name + "_done")
        self.out_path = os.path.join(working_path, job_name)
        self.worker_factory = worker_factory
        exist_or_mkdir(self.mark_path)
        exist_or_mkdir(self.out_path)
        exist_or_mkdir(self.done_path)

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
            self.mark_as_done_job(job_id)
            job_id = mtm.pool_job()
            print("Job id : ", job_id)

    def get_done_path(self, job_id):
        return os.path.join(self.done_path, str(job_id))

    def mark_as_done_job(self, job_id):
        f = open(self.get_done_path(job_id), "w")
        f.write("done")
        f.close()

    def run_one_job(self):
        worker = self.worker_factory(self.out_path)
        worker.work(int(sys.argv[1]))

# Example Usage
#
# if __name__ == "__main__":
#     runner = JobRunner(working_dir, 4000, "unmasked_split", SplitWorker)
#     runner.start()
#

