from dataset_specific.msmarco.common import load_candidate_doc_list_10
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from tlm.data_gen.doc_encode_common import seg_selection_by_geo_sampling
from tlm.data_gen.msmarco_doc_gen.gen_worker_qd import PairwiseQueryDocGenFromTokensList
from tlm.data_gen.msmarco_doc_gen.mmd_gen_common import MMDGenI
from tlm.data_gen.msmarco_doc_gen.mmd_worker import MMDWorker
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceTitleBodyTokensListTrain
from tlm.data_gen.query_document_encoder import QueryDocumentEncoder

if __name__ == "__main__":
    split = "train"
    resource = ProcessedResourceTitleBodyTokensListTrain(split, load_candidate_doc_list_10)
    max_seq_length = 512
    too_short_len = 60
    document_encoder = QueryDocumentEncoder(max_seq_length, True, seg_selection_by_geo_sampling())
    generator: MMDGenI = PairwiseQueryDocGenFromTokensList(resource, document_encoder, max_seq_length)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunnerS(job_man_dir, len(resource.query_group)-1, "MMD_{}_qd".format(split), factory)
    runner.start()
