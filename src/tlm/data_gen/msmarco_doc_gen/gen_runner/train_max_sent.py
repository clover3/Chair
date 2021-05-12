from arg.bm25 import BM25
from cache import load_from_pickle
from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.msmarco_doc_gen.gen_worker import MMDWorker, PointwiseGen
from tlm.data_gen.msmarco_doc_gen.max_sent_encode import MaxSentEncoder, BestSentGen
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource, ProcessedResource10docMulti


def get_bm25_module():
    df = load_from_pickle("mmd_df_10")
    avdl_raw = 1350
    avdl_passage = 40
    # k_dtf_saturation = 1.2
    k_dtf_saturation = 0.75
    return BM25(df, avdl=avdl_passage, num_doc=321384, k1=k_dtf_saturation, k2=100, b=0.75)


if __name__ == "__main__":
    split = "train"
    resource = ProcessedResource10docMulti(split)
    max_seq_length = 512
    basic_encoder = MaxSentEncoder(get_bm25_module(), max_seq_length)
    generator = BestSentGen(resource, basic_encoder, max_seq_length)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunner(job_man_dir, len(resource.query_group)-1,
                       "MMD_train_max_sent".format(split), factory)
    runner.start()
