import os
import pickle

import datastore.tool
from cache import save_to_pickle
from cpath import data_path
from data_generator.tokenizer_wo_tf import FullTokenizer
from galagos.doc_processor import process_jsonl
from misc_lib import file_iterator_interval, exist_or_mkdir


class JsonlWorker:
    def __init__(self, jsonl_path, out_path_not_used):
        voca_path = os.path.join(data_path, "bert_voca.txt")
        self.tokenize_fn = FullTokenizer(voca_path, True).tokenize
        self.jsonl_path = jsonl_path

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


class JsonlWorker2:
    def __init__(self, jsonl_path, out_dir):
        voca_path = os.path.join(data_path, "bert_voca.txt")
        self.tokenize_fn = FullTokenizer(voca_path, True).tokenize
        self.jsonl_path = jsonl_path
        self.out_dir = out_dir
        exist_or_mkdir(out_dir)

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
        pickle_name = "{}.pickle".format(save_name)
        path = os.path.join(self.out_dir, pickle_name)
        pickle.dump(payload_saver, open(path, "wb"))
