from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from tlm.data_gen.doc_encode_common import seg_selection_by_geo_sampling
from tlm.data_gen.msmarco_doc_gen.gen_worker_qd import QueryDocEntityConcatPointwisePredictionGen
from tlm.data_gen.msmarco_doc_gen.mmd_worker import MMDWorker
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceTitleBodyTokensListPredict
from tlm.data_gen.query_document_encoder import QueryDocumentEncoder
from tlm.qtype.qid_to_content_tokens import get_qid_to_content_tokens


def run_for_prediction_split(split):
    resource = ProcessedResourceTitleBodyTokensListPredict(split)
    q_max_seq_length = 128
    max_seq_length = 512
    encoder = QueryDocumentEncoder(max_seq_length, True, seg_selection_by_geo_sampling())
    qid_to_entity_tokens = get_qid_to_content_tokens(split)
    generator = QueryDocEntityConcatPointwisePredictionGen(resource, encoder, max_seq_length,
                                                           q_max_seq_length,
                                                           qid_to_entity_tokens)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunnerS(job_man_dir, len(resource.query_group), "MMD_{}_qe_de".format(split), factory)
    runner.start()


if __name__ == "__main__":
    # split = "dev"
    run_for_prediction_split("dev")
    run_for_prediction_split("test")