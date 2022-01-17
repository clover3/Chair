import json
import os
from typing import List, Tuple, Dict

from cache import load_from_pickle
from cpath import output_path
from data_generator.tokenize_helper import TokenizedText
from data_generator.tokenizer_wo_tf import get_tokenizer
from epath import job_man_dir
from galagos.doc_processor import jsonl_to_tokenized_text
from job_manager.job_runner_with_server import JobRunnerF
from misc_lib import ceil_divide, Averager, file_iterator_interval, exist_or_mkdir


def work(job_no):
    file_path = os.path.join(output_path, "ca_building", "run5", "docs.jsonl")
    save_dir = os.path.join(output_path, "ca_building", "run5", "parsed_doc")
    exist_or_mkdir(save_dir)
    size_per_job = 1000 * 10
    f = open(file_path, "r")
    st = job_no * size_per_job
    ed = st + size_per_job
    iter = file_iterator_interval(f, st, ed)
    print("Job {}".format(job_no))
    output: List[Tuple[str, TokenizedText]] = jsonl_to_tokenized_text(iter, get_tokenizer(), size_per_job)
    output_j: Dict[str, Dict] = {k: v.to_json() for k, v in output}
    save_path = os.path.join(save_dir, "{}.jsonl".format(job_no))
    json.dump(output_j, open(save_path, "w"))


def size_check():
    docs: List[Tuple[str, TokenizedText]] = load_from_pickle("ca_run5_document_processed")
    window_size = 400
    skip = 200
    averager = Averager()
    for doc_id, doc in docs:
        n_window = ceil_divide((doc.get_sb_len() - (window_size-skip)), window_size)
        averager.append(n_window)
    print("Total {} runs, avg {}".format(sum(averager.history), averager.get_average()))


def main():
    num_jobs = 34
    runner = JobRunnerF(job_man_dir, num_jobs, "ca_building_run5_parse_doc", work)
    runner.start()


if __name__ == "__main__":
    main()
