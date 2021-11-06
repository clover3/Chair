from data_generator.tokenizer_wo_tf import get_tokenizer
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from tlm.data_gen.doc_encode_common import seg_selection_by_geo_sampling
from tlm.data_gen.msmarco_doc_gen.gen_qtype.encoders import FromTextEncoderFWDrop, DropTokensDecider
from tlm.data_gen.msmarco_doc_gen.gen_qtype.generator import PairwiseGenWDropTokenFromText
from tlm.data_gen.msmarco_doc_gen.gen_worker import MMDWorker
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource10doc
from tlm.qtype.is_functionword import FunctionWordClassifier


if __name__ == "__main__":
    split = "train"
    resource = ProcessedResource10doc(split)
    max_seq_length = 512
    fw_cls = FunctionWordClassifier()
    tokenizer = get_tokenizer()
    drop_token_decider = DropTokensDecider(fw_cls.is_function_word, tokenizer)
    document_encoder = FromTextEncoderFWDrop(max_seq_length, drop_token_decider, tokenizer, 
                                             True, seg_selection_by_geo_sampling())
    generator = PairwiseGenWDropTokenFromText(resource, document_encoder, max_seq_length)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunnerS(job_man_dir, len(resource.query_group)-1, "MMD_{}_qtype1".format(split), factory)
    runner.start()
