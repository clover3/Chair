import os
import pickle
from typing import List, Tuple

from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from cpath import output_path
from data_generator.job_runner import WorkerInterface
from data_generator.tokenizer_wo_tf import get_tokenizer
from galagos.swtt_processor import jsonl_to_swtt
from job_manager.job_runner_with_server import JobRunnerS
from misc_lib import file_iterator_interval


class Worker(WorkerInterface):
    def __init__(self, json_path, out_dir):
        self.tokenizer = get_tokenizer()
        self.out_dir = out_dir
        self.json_path = json_path

    def work(self, job_id):
        f = open(self.json_path, "r")
        step = 1000
        st = step * job_id
        ed = st + step
        iter = file_iterator_interval(f, st, ed)
        output: List[Tuple[str, SegmentwiseTokenizedText]] = jsonl_to_swtt(iter, self.tokenizer)
        save_path = os.path.join(self.out_dir, str(job_id))
        pickle.dump(output, open(save_path, "wb"))


def main():
    json_path = os.path.join(output_path, "ca_building", "run4", "pc_res.txt.docs.jsonl")
    def factory(out_dir):
        return Worker(json_path, out_dir)

    root_dir = os.path.join(output_path, "ca_building", "run4")

    max_job = 70
    runner = JobRunnerS(root_dir, max_job, "pc_res_doc_jsonl_parsing", factory)
    runner.auto_runner()


if __name__ == "__main__":
    main()
