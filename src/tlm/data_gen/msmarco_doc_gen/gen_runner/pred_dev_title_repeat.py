from typing import List, Dict

from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import LeadingN, AllSegmentAsDoc, AllSegmentRepeatTitle
from tlm.data_gen.msmarco_doc_gen.gen_worker import MMDWorker, PredictionTitleRepeatGen, \
    PredictionAllPassageGenerator
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource, ProcessedResourceTitleBodyPredict

if __name__ == "__main__":
    split = "dev"
    resource = ProcessedResourceTitleBodyPredict(split)
    max_seq_length = 512
    basic_encoder = AllSegmentRepeatTitle(max_seq_length)
    generator = PredictionTitleRepeatGen(resource, basic_encoder, max_seq_length)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunner(job_man_dir, len(resource.query_group)-1, "MMD_pred_title_body".format(split), factory)
    runner.start()
