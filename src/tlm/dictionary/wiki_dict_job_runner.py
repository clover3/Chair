from data_generator import job_runner
import os
from data_generator.common import get_tokenizer
import random
from data_generator.job_runner import JobRunner
from path import output_path


class DictWorker(job_runner.WorkerInterface):
    def __init__(self, out_path, generator):
        self.out_dir = out_path
        self.gen = generator()

    def work(self, job_id):
        doc_id = job_id
        if doc_id > 1000:
            doc_id = doc_id % 1000

        docs = self.gen.load_doc_seg(doc_id)
        output_file = os.path.join(self.out_dir, "{}".format(job_id))
        insts = self.gen.create_instances_from_documents(docs)
        random.shuffle(insts)
        self.gen.write_instances(insts, output_file)


class PDictJobRunner(JobRunner):
    def __init__(self, max_job, job_name, data_generator):
        if os.name == "nt":
            working_path = output_path
        else:
            working_path = "/mnt/nfs/work3/youngwookim/data/bert_tf"

        def worker_factory(out_path):
            return DictWorker(out_path, data_generator)

        super(PDictJobRunner, self).__init__(working_path, max_job, job_name, worker_factory)

