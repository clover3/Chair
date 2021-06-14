from cpath import at_data_dir
from data_generator.job_runner import JobRunner
from dataset_specific.msmarco.common import load_query_group
from epath import job_man_dir
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource10doc

from tlm.data_gen.msmarco_doc_gen.segment_prediction import PassageContainAssureEncoder, SegmentPrediction, MMDWorker
from trec.qrel_parse import load_qrels_structured
from trec.types import QRelsDict

if __name__ == "__main__":
    split = "train"
    resource = ProcessedResource10doc(split)
    max_seq_length = 512
    query_group = load_query_group(split)
    encoder = PassageContainAssureEncoder(max_seq_length)
    generator = SegmentPrediction(resource, encoder, max_seq_length)
    msmarco_passage_qrel_path = at_data_dir("msmarco", "qrels.{}.tsv".format(split))
    passage_qrels: QRelsDict = load_qrels_structured(msmarco_passage_qrel_path)

    def factory(out_dir):
        return MMDWorker(split, generator, out_dir)


    num_jobs = len(query_group)-1
    num_jobs = 40
    runner = JobRunner(job_man_dir, num_jobs, "MMD_gold_segment_prediction_{}".format(split), factory)
    runner.start()
