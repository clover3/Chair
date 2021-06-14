from data_generator.job_runner import JobRunner
from dataset_specific.msmarco.common import load_query_group
from dataset_specific.msmarco.passage_to_doc.resource_loader import load_qrel, load_queries_as_d
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import LeadingN
from tlm.data_gen.msmarco_doc_gen.gen_worker_for_passage_based import PairGenerator, MMDWorkerForPassageBased


if __name__ == "__main__":
    split = "train"
    max_seq_length = 512
    query_group = load_query_group(split)
    qrel = load_qrel(split)
    queries_d = load_queries_as_d(split)
    basic_encoder = LeadingN(max_seq_length, 1)
    generator = PairGenerator(basic_encoder,
                              query_group,
                              qrel,
                              queries_d,
                              max_seq_length)

    def factory(out_dir):
        return MMDWorkerForPassageBased(query_group, generator, out_dir)

    runner = JobRunner(job_man_dir, len(query_group)-1, "MMD_passage_based_{}".format(split), factory)
    runner.start()
