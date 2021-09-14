import os
from collections import Counter
from typing import List

from arg.counter_arg_retrieval.build_dataset.verify_by_acess_log import parse_mturk_time
from contradiction.medical_claims.annotation_1.read_batch import load_file_list
from cpath import output_path
from misc_lib import get_dir_files, group_by
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


def summarize_done_state(file_path_list):
    hit_scheme = HITScheme([], [])

    all_hits = []
    for file_path in file_path_list:
        print(file_path)
        hit_results: List[HitResult] = parse_file(file_path, hit_scheme)
        all_hits.extend(hit_results)

    count_per_hit = Counter()
    for hit in all_hits:
        count_per_hit[hit.hit_id] += 1

    done_distrib = Counter()
    for key, cnt in count_per_hit.items():
        done_distrib[cnt] += 1

    for key, cnt in done_distrib.items():
        print("{} Hits has {} done".format(cnt, key))


def user_check(file_path_list):
    hit_scheme = HITScheme([], [])
    target_id = "A1J1MXAI07HGUT"
    for file_path in file_path_list:
        print(file_path)
        hit_results: List[HitResult] = parse_file(file_path, hit_scheme)

        grouped = group_by(hit_results, lambda x: x.hit_id)
        for key, elems in grouped.items():
            if len(elems) == 3:
                worker_ids = list([e.worker_id for e in elems])
                if target_id not in worker_ids:
                    print(key)


def all_summary():
    dir_path = os.path.join(output_path, "alamri_annotation1", "batch_results")

    workers = summarize_workers(get_dir_files(dir_path))
    workers.sort(key=lambda w: w.n_hit, reverse=True)
    for worker in workers:
        print("{}\t{}\t{}\t{}".format(worker.id, worker.n_hit, worker.first_time, worker.last_time))


def run_summarize_done_state():
    files = load_file_list()
    for file in files:
        summarize_done_state([file])


def main():
    files = load_file_list()
    summarize_done_state(files)


def do_user_check():
    files = load_file_list()
    user_check(files)


def run_per_file():
    dir_path = os.path.join(output_path, "alamri_annotation1", "batch_results")

    for file_path in get_dir_files(dir_path):
        workers = summarize_workers([file_path])
        workers.sort(key=lambda w: w.n_hit, reverse=True)
        for worker in workers:
            print("{}\t{}\t{}\t{}".format(worker.id, worker.n_hit, worker.first_time, worker.last_time))


if __name__ == "__main__":
    main()
