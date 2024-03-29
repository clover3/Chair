import os
from typing import List

from bert_api.task_clients.nli_interface.nli_interface import load_nli_input_from_jsonl, NLIInput, NLIPredictorFromSegTextSig
from bert_api.task_clients.nli_interface.nli_predictors import get_nli_client
from cache import save_list_to_jsonl
from data_generator.job_runner import WorkerInterface
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS


class NLIPayloadWorker(WorkerInterface):
    def __init__(self, client: NLIPredictorFromSegTextSig, input_dir, out_dir):
        self.out_dir = out_dir
        self.input_dir = input_dir
        self.client: NLIPredictorFromSegTextSig = client

    def work(self, job_id):
        payload_path = os.path.join(self.input_dir, str(job_id))
        save_path = os.path.join(self.out_dir, str(job_id))
        items: List[NLIInput] = load_nli_input_from_jsonl(payload_path)
        scores_list: List[List[float]] = self.client(items)
        save_list_to_jsonl(scores_list, save_path)


def main():
    num_jobs = 537
    split = "dev"
    job_name = "alamri_{}_pert_align_eval".format(split)
    payload_job_name = "alamri_{}_perturbation_payloads".format(split)
    payload_dir = os.path.join(job_man_dir, payload_job_name)

    def factory(out_dir):
        client = get_nli_client("direct")
        return NLIPayloadWorker(client, payload_dir, out_dir)

    worker_time_limit = 4 * 3600 - 100

    runner = JobRunnerS(job_man_dir, num_jobs, job_name, factory, worker_time_limit=worker_time_limit)
    runner.start()


if __name__ == "__main__":
    main()