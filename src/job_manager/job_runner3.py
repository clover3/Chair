import itertools
import os
import pickle
from abc import ABC, abstractmethod
from typing import TypeVar, Callable, Iterable, List, NamedTuple, Iterator

from data_generator.job_runner import WorkerInterface
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from misc_lib import ceil_divide

InputV = TypeVar('InputV')
OutputV = TypeVar('OutputV')


# This worker knows nothing about path
class IteratorWorkerSpec(ABC):
    @abstractmethod
    def work(self, data_itr: Iterable) -> List:
        pass


class PartitionSpec(NamedTuple):
    num_record_per_job: int
    num_job: int
    num_record: int

    @classmethod
    def from_total_size(cls, num_record, num_record_per_job):
        num_job = ceil_divide(num_record, num_record_per_job)
        return PartitionSpec(num_record_per_job, num_job, num_record)

    @classmethod
    def from_number_of_jobs(cls, num_jobs, num_record_per_job):
        num_record = num_jobs * num_record_per_job
        return PartitionSpec(num_record_per_job, num_jobs, num_record)


class PartitionDataSpec(NamedTuple):
    num_record_per_job: int
    num_job: int
    num_record: int
    dir_path: str

    @classmethod
    def build(cls, ps: PartitionSpec, dir_path):
        return PartitionDataSpec(ps.num_record_per_job,
                                 ps.num_job,
                                 ps.num_record,
                                 dir_path)

    def get_records_path_for_job(self, job_id):
        return os.path.join(self.dir_path, str(job_id))

    def read_pickles_as_itr(self) -> Iterator:
        for i in range(self.num_job):
            path = self.get_records_path_for_job(i)
            some_list = pickle.load(open(path, "rb"))
            yield from some_list


class IteratorToPickleWorker(WorkerInterface):
    def __init__(self, ps, data_iter_fn: Callable[[], Iterable], worker: IteratorWorkerSpec, out_dir):
        self.save_pds: PartitionDataSpec = PartitionDataSpec.build(ps, out_dir)
        self.data_iter_fn: Callable = data_iter_fn
        self.worker = worker

    def work(self, job_id):
        ps = self.save_pds
        st = ps.num_record_per_job * job_id
        ed = st + ps.num_record_per_job
        data_iter = self.data_iter_fn()
        todo: Iterable = itertools.islice(data_iter, st, ed)
        out_data: List = self.worker.work(todo)
        save_path = self.save_pds.get_records_path_for_job(job_id)
        pickle.dump(out_data, open(save_path, "wb"))


# Input: data_iter
# Output: List
# saves pickle at "job_man_dir/job_name/job_id"

def run_iterator_to_pickle_worker(ps: PartitionSpec, data_iter_fn: Callable[[], Iterator],
                                  worker: IteratorWorkerSpec, job_name):
    print("Run IteratorToPickleWorker. job_name={}".format(job_name))
    # Let's define canonical WorkerInterface

    def worker_factory(out_dir):
        return IteratorToPickleWorker(ps, data_iter_fn, worker, out_dir)

    job_runner_s = JobRunnerS(job_man_dir, ps.num_job, job_name, worker_factory)
    job_runner_s.start()

