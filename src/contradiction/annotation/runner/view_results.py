import os
from typing import List

from arg.counter_arg_retrieval.build_dataset.verify_by_acess_log import parse_mturk_time
from cpath import output_path
from misc_lib import get_dir_files
from mturk.parse_util import HITScheme, HitResult, parse_file


class WorkerInfo:
    def __init__(self, id):
        self.id = id
        self.n_hit = 0
        self.first_time = None
        self.last_time = None

    def update(self, hit: HitResult):
        self.n_hit += 1
        submit_time_parsed = parse_mturk_time(hit.submit_time)
        if self.first_time is None or submit_time_parsed < self.first_time:
            self.first_time = submit_time_parsed

        if self.last_time is None or submit_time_parsed > self.last_time:
            self.last_time = submit_time_parsed


def summarize_workers(file_path_list):
    print(file_path_list)
    hit_scheme = HITScheme([], [])

    all_hits = []
    for file_path in file_path_list:
        print(file_path)
        hit_results: List[HitResult] = parse_file(file_path, hit_scheme)
        all_hits.extend(hit_results)

    workers = {}
    for h in all_hits:
        key = h.worker_id
        if key not in workers:
            workers[key] = WorkerInfo(key)
        workers[key].update(h)

    return list(workers.values())


def all_summary():
    dir_path = os.path.join(output_path, "alamri_annotation1", "batch_results")

    workers = summarize_workers(get_dir_files(dir_path))
    workers.sort(key=lambda w: w.n_hit, reverse=True)
    for worker in workers:
        print("{}\t{}\t{}\t{}".format(worker.id, worker.n_hit, worker.first_time, worker.last_time))


def main():
    all_summary()


def run_per_file():
    dir_path = os.path.join(output_path, "alamri_annotation1", "batch_results")

    for file_path in get_dir_files(dir_path):
        workers = summarize_workers([file_path])
        workers.sort(key=lambda w: w.n_hit, reverse=True)
        for worker in workers:
            print("{}\t{}\t{}\t{}".format(worker.id, worker.n_hit, worker.first_time, worker.last_time))


if __name__ == "__main__":
    run_per_file()
