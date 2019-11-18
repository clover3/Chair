from dictionary.reader import DictionaryReader
from tlm.data_gen.dict_reader import dictionary_encoder
from data_generator.common import get_tokenizer
import pickle
from cache import save_to_pickle
import random
from cache import *
import sys
from sydney_manager import MarkedTaskManager
from misc_lib import TimeEstimator
from tlm.data_gen.dict_reader import DictTrainGen, Dictionary
from tlm.data_gen import run_unmasked_pair_gen
from tlm.tf_logging import tf_logging
import logging

working_path ="/mnt/nfs/work3/youngwookim/data/bert_tf"

def encode_dictionary():
    path = "c:\\work\\Data\\webster\\webster_headerless.txt"
    d1 = DictionaryReader.open(path)
    d = dictionary_encoder(d1.entries, get_tokenizer())
    save_to_pickle(d, "webster")

class DGenWorker(run_unmasked_pair_gen.Worker):
    def __init__(self, out_path):
        super(DGenWorker, self).__init__(out_path)
        self.out_dir = out_path
        d = Dictionary(load_from_pickle("webster"))
        self.gen = DictTrainGen(d)

def main():
    mark_path = os.path.join(working_path, "dict_reader_mark")
    out_path = os.path.join(working_path, "dict_reader")
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    mtm = MarkedTaskManager(4000, mark_path, 1)
    worker = DGenWorker(out_path)

    job_id = mtm.pool_job()
    print("Job id : ", job_id)
    while job_id is not None:
        worker.work(job_id)
        job_id = mtm.pool_job()
        print("Job id : ", job_id)

def simple():
    out_path = os.path.join(working_path, "dict_reader")
    worker = DGenWorker(out_path)
    worker.work(int(sys.argv[1]))

if __name__ == "__main__":
    main()


