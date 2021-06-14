from data_generator.job_runner import JobRunner
from dataset_specific.msmarco.common import load_candidate_doc_list_10
from epath import job_man_dir
from tlm.data_gen.adhoc_sent_tokenize import FromSentTokensListEncoder
from tlm.data_gen.doc_encode_common import seg_selection_by_geo_sampling
from tlm.data_gen.msmarco_doc_gen.gen_worker import MMDWorker
from tlm.data_gen.msmarco_doc_gen.gen_worker_from_tokens_list import PointwiseGenFromTokensList
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceTitleBodyTokensListTrain

if __name__ == "__main__":
    split = "dev"
    resource = ProcessedResourceTitleBodyTokensListTrain(split, load_candidate_doc_list_10)
    max_seq_length = 512
    document_encoder = FromSentTokensListEncoder(max_seq_length, True, seg_selection_by_geo_sampling())
    generator = PointwiseGenFromTokensList(resource, document_encoder, max_seq_length)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunner(job_man_dir, len(resource.query_group)-1, "MMD_{}_D".format(split), factory)
    runner.start()
