import logging
import random
import sys

from cache import *
from misc_lib import exist_or_mkdir
from sydney_manager import MarkedTaskManager
from tlm.dictionary.data_gen import DictLookupPredictGen
from tlm.tf_logging import tf_logging

working_path ="/mnt/nfs/work3/youngwookim/data/bert_tf"
from tlm.dictionary.data_gen import Dictionary


class Worker:
    def __init__(self, example_out_path, key_out_path, n_out_path):
        self.example_out_dir = example_out_path
        self.key_out_dir = key_out_path
        self.n_out_path = n_out_path
        d = Dictionary(load_from_pickle("webster"))
        self.gen = DictLookupPredictGen(d, samples_n=10)


    def work(self, job_id):
        doc_id = job_id
        if doc_id >= 1000:
            doc_id = doc_id % 1000

        docs = self.gen.load_doc_seg(doc_id)
        example_file = os.path.join(self.example_out_dir, "{}".format(job_id))
        key_file = os.path.join(self.key_out_dir, "{}".format(job_id))
        insts = self.gen.create_instances_from_documents(docs)
        random.shuffle(insts)
        n_list = self.gen.write_instances(insts, example_file, key_file)

        n_out_path = os.path.join(self.n_out_path, "{}".format(job_id))
        f = open(n_out_path, "w")
        for n in n_list:
            f.write("{}\n".format(n))
        f.close()

def init_worker():
    out_path1 = os.path.join(working_path, "lookup_example")
    out_path2 = os.path.join(working_path, "lookup_key")
    out_path3 = os.path.join(working_path, "lookup_n")
    exist_or_mkdir(out_path1)
    exist_or_mkdir(out_path2)
    exist_or_mkdir(out_path3)

    worker = Worker(out_path1, out_path2, out_path3)
    return worker


def main():
    mark_path = os.path.join(working_path, "lookup_mark")
    mtm = MarkedTaskManager(4000, mark_path, 1)

    worker = init_worker()
    job_id = mtm.pool_job()
    print("Job id : ", job_id)
    while job_id is not None:
        worker.work(job_id)
        job_id = mtm.pool_job()
        print("Job id : ", job_id)


def simple():
    tf_logging.setLevel(logging.INFO)
    worker = init_worker()
    worker.work(int(sys.argv[1]))


if __name__ == "__main__":
    simple()

