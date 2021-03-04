from typing import List, Dict

from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import LeadingN
from tlm.data_gen.msmarco_doc_gen.gen_worker import MMDWorker, ProcessedResource, FirstPassageGenerator

if __name__ == "__main__":
    split = "train"
    resource = ProcessedResource(split)
    max_seq_length = 512
    basic_encoder = LeadingN(max_seq_length, 1)
    generator = FirstPassageGenerator(resource, basic_encoder, max_seq_length)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunner(job_man_dir, len(resource.query_group)-1, "MMD_train".format(split), factory)
    runner.start()
