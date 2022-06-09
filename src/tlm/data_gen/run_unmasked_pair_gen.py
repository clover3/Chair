import random
import sys

from cache import *
from job_manager.marked_task_manager import MarkedTaskManager
from tlm.data_gen.lm_datagen import UnmaskedPairGen

working_path ="/mnt/nfs/work3/youngwookim/data/bert_tf"


class Worker:
    def __init__(self, out_path):
        self.out_dir = out_path
        self.gen = UnmaskedPairGen()

    def work(self, job_id):
        doc_id = job_id
        if doc_id > 1000:
            doc_id = doc_id % 1000

        docs = self.gen.load_doc_seg(doc_id)
        output_file = os.path.join(self.out_dir, "{}".format(job_id))
        insts = self.gen.create_instances_from_documents(docs)
        random.shuffle(insts)
        self.gen.write_instance_to_example_files(insts, [output_file])

def main():
    mark_path = os.path.join(working_path, "unmasked_pair_x3_mark")
    out_path = os.path.join(working_path, "unmasked_pair_x3")
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    mtm = MarkedTaskManager(4000, mark_path, 1)
    worker = Worker(out_path)

    job_id = mtm.pool_job()
    print("Job id : ", job_id)
    while job_id is not None:
        worker.work(job_id)
        job_id = mtm.pool_job()
        print("Job id : ", job_id)


def simple():
    out_path = os.path.join(working_path, "tf_unmasked")
    worker = Worker(out_path)
    worker.work(int(sys.argv[1]))


if __name__ == "__main__":
    main()

