from data_generator.job_runner import JobRunner
from dataset_specific.msmarco.common import load_query_group
from epath import job_man_dir
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourcePredict

from tlm.data_gen.msmarco_doc_gen.segment_prediction import PassageContainAssureEncoder, SegmentPrediction, MMDWorker

if __name__ == "__main__":
    split = "dev"
    query_group = load_query_group(split)
    resource = ProcessedResourcePredict(split)
    max_seq_length = 512
    encoder = PassageContainAssureEncoder(max_seq_length)
    generator = SegmentPrediction(resource, encoder, max_seq_length)

    def factory(out_dir):
        return MMDWorker(split, generator, out_dir)

    runner = JobRunner(job_man_dir, len(query_group)-1, "MMD_gold_segment_prediction_{}".format(split), factory)
    runner.start()
