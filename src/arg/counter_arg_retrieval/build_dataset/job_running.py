import json
import os

from arg.counter_arg_retrieval.build_dataset.passage_scoring.passage_scoring import scoring_output_to_json
from cpath import output_path
from data_generator.job_runner import WorkerInterface, ListWorker
from job_manager.job_runner_with_server import JobRunnerS


class ListJson(WorkerInterface):
    def __init__(self, work_fn, todo, out_dir):
        self.work_fn = work_fn
        self.todo = todo
        self.out_dir = out_dir

    def work(self, job_id):
        save_path = os.path.join(self.out_dir, "{}.json".format(str(job_id)))
        output = self.work_fn(self.todo[job_id:job_id+1])
        j = scoring_output_to_json(output)
        json.dump(j, open(save_path, "w"), indent=True)


def run_job_runner(ca_query_list, work_fn, job_name):
    def factory(out_dir):
        return ListWorker(work_fn, ca_query_list, out_dir)

    root_dir = os.path.join(output_path, "ca_building", "jobs")
    job_runner = JobRunnerS(root_dir, len(ca_query_list), job_name, factory)
    job_runner.auto_runner()


def run_job_runner_json(ca_query_list, work_fn, job_name):
    def factory(out_dir):
        return ListJson(work_fn, ca_query_list, out_dir)

    root_dir = os.path.join(output_path, "ca_building", "jobs")
    job_runner = JobRunnerS(root_dir, len(ca_query_list), job_name, factory)
    job_runner.auto_runner()