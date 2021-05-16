import os
from typing import List, Dict

from data_generator.job_runner import JobRunner, WorkerInterface
from epath import job_man_dir
from misc_lib import exist_or_mkdir
from tlm.data_gen.adhoc_datagen import MultiWindow
from tlm.data_gen.adhoc_sent_tokenize import SeroFromTextEncoder
from tlm.data_gen.msmarco_doc_gen.fix_train_sero_A import do_fix
from tlm.data_gen.msmarco_doc_gen.gen_worker import MMDWorker
from tlm.data_gen.msmarco_doc_gen.gen_worker_sent_level import PointwiseGenFromText
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource10doc


class FixWorker(WorkerInterface):
    def __init__(self, in_dir, out_dir):
        self.out_dir = out_dir
        self.in_dir = in_dir
        exist_or_mkdir(self.out_dir)

    def work(self, job_id):
        do_fix(
            os.path.join(self.in_dir, str(job_id)),
            os.path.join(self.out_dir, str(job_id))
        )


if __name__ == "__main__":
    split = "train"
    resource = ProcessedResource10doc(split)
    in_dir = os.path.join(job_man_dir, "MMD_pair_512_4")
    def factory(out_dir):
        return FixWorker(in_dir, out_dir)
    runner = JobRunner(job_man_dir, len(resource.query_group)-1, "MMD_pair_512_4_fix", factory)
    runner.start()
