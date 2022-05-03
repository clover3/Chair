import os
from typing import List

from bert_api.task_clients.nli_interface.nli_interface import load_nli_input_from_jsonl, NLIInput
from bert_api.task_clients.nli_interface.nli_predictors_path import get_nli_cache_sqlite_path
from cache import load_list_from_jsonl
from cpath import output_path
from data_generator.job_runner import WorkerInterface
from misc_lib import validate_equal
from tlm.qtype.partial_relevance.cache_db import bulk_save_s


class Worker(WorkerInterface):
    def __init__(self, input_dir, out_dir, sql_path):
        self.out_dir = out_dir
        self.input_dir = input_dir
        self.sql_path = sql_path

    def work(self, job_id):
        payload_path = os.path.join(self.input_dir, str(job_id))
        save_path = os.path.join(self.out_dir, str(job_id))
        items: List[NLIInput] = load_nli_input_from_jsonl(payload_path)
        scores_list = load_list_from_jsonl(save_path, lambda j: j)
        validate_equal(len(items), len(scores_list))
        key_list = map(NLIInput.str_hash, items)
        key_value_list = dict(zip(key_list, scores_list))
        bulk_save_s(self.sql_path, key_value_list)


def main():
    num_jobs = 537
    split = "dev"
    job_name = "alamri_{}_pert_align_eval".format(split)
    payload_job_name = "alamri_{}_perturbation_payloads".format(split)
    payload_dir = os.path.join(output_path, payload_job_name)
    prediction_path = os.path.join(output_path, job_name)

    worker = Worker(payload_dir, prediction_path, get_nli_cache_sqlite_path())

    for i in range(num_jobs):
        print("job {}".format(i))
        worker.work(i)


def debug():
    num_jobs = 537
    split = "dev"
    job_name = "alamri_{}_pert_align_eval".format(split)
    payload_job_name = "alamri_{}_perturbation_payloads".format(split)
    payload_dir = os.path.join(output_path, payload_job_name)
    prediction_path = os.path.join(output_path, job_name)

    for job_id in range(num_jobs):
        payload_path = os.path.join(payload_dir, str(job_id))
        save_path = os.path.join(prediction_path, str(job_id))
        items: List[NLIInput] = load_nli_input_from_jsonl(payload_path)
        scores_list = load_list_from_jsonl(save_path, lambda j: j)
        validate_equal(len(items), len(scores_list))
        print("job {}: {} records".format(job_id, len(scores_list)))


if __name__ == "__main__":
    main()