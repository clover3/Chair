from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.common import load_candidate_doc_list_10
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from tlm.data_gen.doc_encode_common import seg_selection_by_geo_sampling
from tlm.data_gen.msmarco_doc_gen.gen_worker_qd import QueryDocEntityConcatGen
from tlm.data_gen.msmarco_doc_gen.mmd_worker import MMDWorker
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceTitleBodyTokensListTrain
from tlm.data_gen.query_document_encoder import QueryDocumentEncoder
from tlm.qtype.qid_to_content_tokens import get_qid_to_content_tokens

if __name__ == "__main__":
    split = "train"
    resource = ProcessedResourceTitleBodyTokensListTrain(split, load_candidate_doc_list_10)
    q_max_seq_length = 128
    max_seq_length = 512
    tokenizer = get_tokenizer()
    encoder = QueryDocumentEncoder(max_seq_length, True, seg_selection_by_geo_sampling())
    qid_to_entity_tokens = get_qid_to_content_tokens(split)
    generator = QueryDocEntityConcatGen(resource, encoder, max_seq_length,
                                        q_max_seq_length,
                                        qid_to_entity_tokens)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunnerS(job_man_dir, len(resource.query_group)-1, "MMD_{}_qe_de".format(split), factory)
    runner.start()
