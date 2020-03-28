import os

import datastore.tool
from cache import save_to_pickle
from cpath import data_path
from data_generator.job_runner import JobRunner, sydney_working_dir
from data_generator.tokenizer_wo_tf import FullTokenizer
from galagos.doc_processor import process_jsonl
###
from misc_lib import file_iterator_interval


class JsonlWorker:
    def __init__(self, out_path_not_used):
        voca_path = os.path.join(data_path, "bert_voca.txt")
        self.tokenize_fn = FullTokenizer(voca_path, True).tokenize
        #self.jsonl_path = "/mnt/nfs/work3/youngwookim/data/perspective/train_claim_perspective/docs_BM25_100.jsonl"
        self.jsonl_path = "/mnt/nfs/work3/youngwookim/data/perspective/dev_claim_perspective/docs.jsonl"

    def work(self, job_id):
        jsonl_path = self.jsonl_path
        f = open(jsonl_path, "r")
        block = 1000
        st = job_id * block
        ed = (job_id + 1) * block
        line_itr = file_iterator_interval(f, st, ed)
        buffered_saver = datastore.tool.PayloadSaver()
        payload_saver = process_jsonl(line_itr, self.tokenize_fn, buffered_saver)
        save_name = os.path.basename(jsonl_path) + "_{}".format(job_id)
        save_to_pickle(payload_saver, save_name)


num_dev_pc_jos = 238

if __name__ == "__main__":
    runner = JobRunner(sydney_working_dir, num_dev_pc_jos, "pc_dev", JsonlWorker)
    runner.start()


