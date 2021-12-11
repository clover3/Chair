from cache import load_from_pickle
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from tlm.data_gen.adhoc_datagen import FirstAndTitle
from tlm.data_gen.msmarco_doc_gen.gen_worker import GenerateFromTitleBody2
from tlm.data_gen.msmarco_doc_gen.gen_worker_qd import RandomDropGenerator
from tlm.data_gen.msmarco_doc_gen.mmd_worker import MMDWorker
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceTitleBodyCorpusWise
from tlm.qtype.content_functional_parsing.derived_query_set import load_derived_query_set_a
from tlm.qtype.qde_resource import QDEResourceFlat


def main():
    split = "train"
    query_set = load_derived_query_set_a(split)
    candidate_docs_d = load_from_pickle("MMD_candidate_docs_d_{}".format(split))
    resource_source = ProcessedResourceTitleBodyCorpusWise(candidate_docs_d, split)
    resource = QDEResourceFlat(resource_source, query_set)

    max_seq_length = 512
    basic_encoder = FirstAndTitle(max_seq_length)
    skip_rate = 0.75
    generator = GenerateFromTitleBody2(resource, basic_encoder, max_seq_length, skip_rate)
    random_drop_gen = RandomDropGenerator(generator, skip_rate)

    def factory(out_dir):
        return MMDWorker(resource.query_group, random_drop_gen, out_dir)

    runner = JobRunnerS(job_man_dir, len(resource.query_group), "MMD_{}_set_a".format(split), factory)
    runner.start()


if __name__ == "__main__":
    main()
