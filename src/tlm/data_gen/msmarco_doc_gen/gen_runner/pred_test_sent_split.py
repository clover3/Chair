from typing import List, Dict

from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import LeadingN, FirstAndRandom, GeoSampler
from tlm.data_gen.adhoc_sent_tokenize import FromTextEncoder
from tlm.data_gen.doc_encode_common import seg_selection_by_geo_sampling
from tlm.data_gen.msmarco_doc_gen.gen_worker import MMDWorker
from tlm.data_gen.msmarco_doc_gen.gen_worker_sent_level import PredictionGen
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource, ProcessedResource10doc, \
    ProcessedResourcePredict


def get_first_ten(items):
    for idx, item in enumerate(items):
        if idx < 10:
            yield


if __name__ == "__main__":
    split = "test"
    resource = ProcessedResourcePredict(split)
    max_seq_length = 512
    document_encoder = FromTextEncoder(max_seq_length,
                                       random_short=True,
                                       seg_selection_fn=None,
                                       max_seg_per_doc=20)
    generator = PredictionGen(resource, document_encoder, max_seq_length)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunner(job_man_dir, len(resource.query_group)-1, "MMD_{}_sent_split".format(split), factory)
    runner.start()
