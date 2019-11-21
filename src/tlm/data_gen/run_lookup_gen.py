import random
from cache import *
from path import data_path
from data_generator import tokenizer_wo_tf as tokenization
import sys
from sydney_manager import MarkedTaskManager
from misc_lib import TimeEstimator,exist_or_mkdir
from tlm.data_gen.dict_reader import DictLookupPredcitGen
import time
working_path ="/mnt/nfs/work3/youngwookim/data/bert_tf"
from tlm.data_gen.dict_reader import DictTrainGen, Dictionary


class Worker:
    def __init__(self, example_out_path, key_out_path):
        self.example_out_dir = example_out_path
        self.key_out_dir = key_out_path
        d = Dictionary(load_from_pickle("webster"))
        self.gen = DictLookupPredcitGen(d, samples_n=10)


    def work(self, job_id):
        doc_id = job_id
        if doc_id >= 1000:
            doc_id = doc_id % 1000

        docs = self.gen.load_doc_seg(doc_id)
        example_file = os.path.join(self.example_out_dir, "{}".format(job_id))
        key_file = os.path.join(self.key_out_dir, "{}".format(job_id))
        insts = self.gen.create_instances_from_documents(docs)
        random.shuffle(insts)
        self.gen.write_instances(insts, example_file, key_file)

def main():
    mark_path = os.path.join(working_path, "lookup_mark")
    out_path1 = os.path.join(working_path, "lookup_example")
    out_path2 = os.path.join(working_path, "lookup_key")
    exist_or_mkdir(out_path1)
    exist_or_mkdir(out_path2)

    mtm = MarkedTaskManager(4000, mark_path, 1)
    worker = Worker(out_path1, out_path2)

    job_id = mtm.pool_job()
    print("Job id : ", job_id)
    while job_id is not None:
        worker.work(job_id)
        job_id = mtm.pool_job()
        print("Job id : ", job_id)


def simple():
    out_path1 = os.path.join(working_path, "lookup_example")
    out_path2 = os.path.join(working_path, "lookup_key")
    worker = Worker(out_path1, out_path2)
    worker.work(int(sys.argv[1]))


if __name__ == "__main__":
    simple()

