

from typing import List, Dict

from data_generator.job_runner import JobRunner
from dataset_specific.msmarco.common import load_candidate_doc_list_10
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import FirstNoTitle, AllSegmentNoTitle
from tlm.data_gen.msmarco_doc_gen.gen_worker import MMDWorker, PredictionGenFromTitleBody, \
    PointwiseGen, PredictionGenFromTitleBody
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceTitleBodyTrain, \
    ProcessedResourceTitleBodyPredict

if __name__ == "__main__":
    split = "dev"
    resource = ProcessedResourceTitleBodyPredict(split)
    max_seq_length = 512
    doc_rep_encoder = AllSegmentNoTitle(max_seq_length)
    generator = PredictionGenFromTitleBody(resource, doc_rep_encoder, max_seq_length)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunner(job_man_dir, len(resource.query_group)-1, "MMD_{}_pred_no_title".format(split), factory)
    runner.start()