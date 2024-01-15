import itertools
import os
import pickle

from data_generator.job_runner import JobRunner, sydney_working_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import TimeEstimator


class DoNotNeed(Exception):
    pass


def _read_lines_st_ed(file_path, st, ed):
    f = open(file_path, "r")
    for line in itertools.islice(f, st, ed):
        yield line


lines_per_job = 1000 * 1000


def do_tokenize(job_id, file_path, total_lines):
    st = lines_per_job * job_id
    ed = lines_per_job * (job_id + 1)
    tokenizer = get_tokenizer()

    if st >= total_lines:
        print("Starting line exceed file length")
        raise DoNotNeed

    ticker = TimeEstimator(lines_per_job)
    tokenized_lines = []
    for line in _read_lines_st_ed(file_path, st, ed):
        tokenized_lines.append(tokenizer.tokenize(line))
        ticker.tick()
    return tokenized_lines


def do_second_file(job_id):
    file_path = "/mnt/nfs/work3/youngwookim/data/books_large_p1.txt"
    total_lines = 40 * 1000 * 1000
    do_tokenize(job_id, file_path, total_lines)


def work(job_id, input_path, save_format, total_lines):
    try:
        tokenized_lines = do_tokenize(job_id, input_path, total_lines)
        save_path = save_format.format(job_id)
        pickle.dump(tokenized_lines, open(save_path, "wb"))
    except DoNotNeed as e:
        return

class FirstFileWorker:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def work(self, job_id):
        file_path = "/mnt/nfs/work3/youngwookim/data/books_large_p1.txt"
        total_lines = 40 * 1000 * 1000
        save_format = os.path.join(self.output_dir, "1_{}")
        work(job_id, file_path, save_format, total_lines)


class SecondFileWorker:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def work(self, job_id):
        file_path = "/mnt/nfs/work3/youngwookim/data/books_large_p2.txt"
        total_lines = 34004228
        save_format = os.path.join(self.output_dir, "2_{}")
        work(job_id, file_path, save_format, total_lines)


if __name__ == "__main__":
    runner = JobRunner(sydney_working_dir, 40, "bookcorpus_tokens", FirstFileWorker)
    runner.start()
    runner = JobRunner(sydney_working_dir, 40, "bookcorpus_tokens2", SecondFileWorker)
    runner.start()

