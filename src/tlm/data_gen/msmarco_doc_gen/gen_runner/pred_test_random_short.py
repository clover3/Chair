from typing import List, Dict

from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import LeadingN, FirstAndRandom, GeoSampler, LeadingNWithRandomShort
from tlm.data_gen.adhoc_sent_tokenize import FromTextEncoder
from tlm.data_gen.doc_encode_common import seg_selection_by_geo_sampling
from tlm.data_gen.msmarco_doc_gen.gen_worker import MMDWorker, PredictionAllPassageGenerator
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource, ProcessedResource10doc, \
    ProcessedResourcePredict


def gen_for_split(split):
    resource = ProcessedResourcePredict(split)
    max_seq_length = 512
    basic_encoder = LeadingNWithRandomShort(max_seq_length, 20)
    generator = PredictionAllPassageGenerator(resource, basic_encoder, max_seq_length)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunner(job_man_dir, len(resource.query_group) - 1, "MMD_pred_{}_random_short".format(split), factory)
    runner.start()


if __name__ == "__main__":
    gen_for_split("test")
    gen_for_split("dev")
