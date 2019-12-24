import os
import pickle

from data_generator.job_runner import JobRunner
from tlm.data_gen.base import UnmaskedGen

working_path ="/mnt/nfs/work3/youngwookim/data/bert_tf"

class Worker:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.unmaskedgen = UnmaskedGen()

    def work(self, job_id):
        doc_id = job_id
        file_path = "/mnt/nfs/work3/youngwookim/data/enwiki4bert/tokens/enwiki_eval_tokens.{}".format(doc_id)
        f = open(file_path, "rb")
        docs = pickle.load(f)
        output_file = os.path.join(self.out_dir, "{}".format(job_id))
        all_insts = []
        for doc in docs:
            insts = self.unmaskedgen.create_instances_from_document(doc)
            all_insts.extend(insts)

        self.unmaskedgen.write_instance_to_example_files(all_insts, [output_file])

if __name__ == "__main__":
    runner = JobRunner(working_path, 100, "unmasked_dev", Worker)
    runner.start()
