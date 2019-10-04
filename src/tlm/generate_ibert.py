import random
from cache import *
from path import data_path
from data_generator import tokenizer_wo_tf as tokenization
import sys
from sydney_manager import MarkedTaskManager
from misc_lib import flatten
from tlm.retreive_candidates import get_visible
from tlm.stem import CacheStemmer
from tlm.galago_query_maker import clean_query
from collections import Counter
import collections
from adhoc.bm25 import BM25_3, BM25_3_q_weight
from misc_lib import left, TimeEstimator
from models.classic.stopword import load_stopwords
from adhoc.galago import load_df, write_query_json
import tensorflow as tf
from tlm.wiki import bert_training_data as btd
from tlm.wiki.bert_training_data import *
import time
from tlm.ibert_data_transformer import convert_write,read_bert_data

working_path ="/mnt/nfs/work3/youngwookim/data/ibert_tf"

class Worker:
    def __init__(self, out_path):
        self.out_path = out_path

    def work(self, job_id):
        path = "/mnt/nfs/work3/youngwookim/data/bert_tf/tf/done/{}".format(job_id)
        data = read_bert_data(path)
        out_path = os.path.join(self.out_path, str(job_id))
        convert_write(out_path, data)


def main():
    mark_path = os.path.join(working_path, "mark")
    out_path = os.path.join(working_path, "tf")
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    mtm = MarkedTaskManager(1, mark_path, 1)
    worker = Worker(out_path)
    job_id = mtm.pool_job()
    print("Job id : ", job_id)
    while job_id is not None:
        worker.work(job_id)
        job_id = mtm.pool_job()
        print("Job id : ", job_id)

def simple():
    out_path = os.path.join(working_path, "tf")
    worker = Worker(out_path)

    worker.work(int(sys.argv[1]))


if __name__ == "__main__":
    main()

