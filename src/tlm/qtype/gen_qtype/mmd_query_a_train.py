from tlm.data_gen.adhoc_datagen import FirstAndTitle
from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import FirstAndTitle
from tlm.data_gen.msmarco_doc_gen.gen_worker import GenerateFromTitleBody
from tlm.data_gen.msmarco_doc_gen.mmd_worker import MMDWorker
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceTitleBodyPredict
from tlm.qtype.content_functional_parsing.derived_query_set import load_derived_query_set_a
from tlm.qtype.qde_resource import QDEResourceFlat

if __name__ == "__main__":
    split = "train"
    query_set = load_derived_query_set_a(split)
    resource_source = ProcessedResourceTitleBodyPredict(split)
    resource = QDEResourceFlat(resource_source, query_set)

    max_seq_length = 512
    basic_encoder = FirstAndTitle(max_seq_length)
    generator = GenerateFromTitleBody(resource, basic_encoder, max_seq_length)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunner(job_man_dir, len(resource.query_group)-1, "MMD_{}_set_a".format(split), factory)
    runner.start()
