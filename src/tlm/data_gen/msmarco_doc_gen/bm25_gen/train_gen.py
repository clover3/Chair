


from data_generator.job_runner import JobRunner
from dataset_specific.msmarco.common import load_token_d_bm25_tokenize, load_token_d_sent_level
from epath import job_man_dir
from tlm.data_gen.msmarco_doc_gen.bm25_gen.bm25_gen_common import BM25SelectedSegmentEncoder
from tlm.data_gen.msmarco_doc_gen.bm25_gen.pairwise_gen import PairGeneratorFromMultiResource
from tlm.data_gen.msmarco_doc_gen.gen_worker import MMDWorker
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource10docMulti

if __name__ == "__main__":
    split = "train"
    resource = ProcessedResource10docMulti(split)
    max_seq_length = 512
    bm25_based_selected_encoder = BM25SelectedSegmentEncoder(max_seq_length)
    def get_tokens_d_bert(qid):
        return load_token_d_sent_level(split, qid)

    def get_tokens_d_bm25(qid):
        return load_token_d_bm25_tokenize(split, qid)
    generator = PairGeneratorFromMultiResource(resource,
                                               resource.get_bert_tokens_d,
                                               resource.get_stemmed_tokens_d,
                                               bm25_based_selected_encoder,
                                               max_seq_length
                                               )

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunner(job_man_dir, len(resource.query_group)-1, "MMD_pair_bm25_sel_{}".format(split), factory)
    runner.start()
