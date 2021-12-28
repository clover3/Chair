from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from tlm.data_gen.adhoc_sent_tokenize import FromSentTokensListEncoder
from tlm.data_gen.doc_encode_common import seg_selection_take_first
from tlm.data_gen.msmarco_doc_gen.gen_worker import GenFromTitleBodyTokensList
from tlm.data_gen.msmarco_doc_gen.mmd_worker import MMDWorker
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceTitleBodyTokensListPredict
from tlm.qtype.content_functional_parsing.derived_query_set import load_derived_query_set_a_small
from tlm.qtype.qde_resource import QDEResource


def run_for_prediction_split(split):
    query_set = load_derived_query_set_a_small(split)
    resource_source = ProcessedResourceTitleBodyTokensListPredict(split)
    resource = QDEResource(resource_source, query_set)
    max_seq_length = 512
    basic_encoder = FromSentTokensListEncoder(max_seq_length, True, seg_selection_take_first())
    generator = GenFromTitleBodyTokensList(resource, basic_encoder, max_seq_length)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunnerS(job_man_dir, len(resource.query_group), "MMD_{}_qd_a_small".format(split), factory)
    runner.start()


if __name__ == "__main__":
    run_for_prediction_split("dev")
    # run_for_prediction_split("test")
