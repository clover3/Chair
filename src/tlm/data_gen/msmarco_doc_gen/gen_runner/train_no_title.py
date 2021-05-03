from typing import List, Dict

from data_generator.job_runner import JobRunner
from dataset_specific.msmarco.common import load_candidate_doc_list_10
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import FirstNoTitle, FirstAndRandomNoTitle
from tlm.data_gen.msmarco_doc_gen.gen_worker import MMDWorker, PredictionGenFromTitleBody, \
    PointwiseGen, GenerateFromTitleBody
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceTitleBodyTrain

if __name__ == "__main__":
    split = "train"
    resource = ProcessedResourceTitleBodyTrain(split, load_candidate_doc_list_10)
    max_seq_length = 512
    doc_rep_encoder = FirstAndRandomNoTitle(max_seq_length)
    generator = GenerateFromTitleBody(resource, doc_rep_encoder, max_seq_length)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunner(job_man_dir, len(resource.query_group)-1, "MMD_{}_no_title".format(split), factory)
    runner.start()