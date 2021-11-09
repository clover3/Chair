from arg.bm25 import BM25
from cache import load_from_pickle
from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.msmarco_doc_gen.max_sent_encode import MaxSentEncoder, BestSentGen
from tlm.data_gen.msmarco_doc_gen.mmd_worker import MMDWorker
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource100docMulti


def get_bm25_module():
    df = load_from_pickle("mmd_df_10")
    avdl_raw = 1350
    avdl_passage = 40
    # k_dtf_saturation = 1.2
    k_dtf_saturation = 0.75
    return BM25(df, avdl=avdl_passage, num_doc=321384, k1=k_dtf_saturation, k2=100, b=0.75)


def work_for(split):
    resource = ProcessedResource100docMulti(split)
    max_seq_length = 512
    basic_encoder = MaxSentEncoder(get_bm25_module(), max_seq_length, True)
    generator = BestSentGen(resource, basic_encoder, max_seq_length)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunner(job_man_dir, len(resource.query_group)-1,
                       "MMD_max_sent_{}_B".format(split), factory)
    runner.start()


if __name__ == "__main__":
    work_for("dev")
    work_for("test")