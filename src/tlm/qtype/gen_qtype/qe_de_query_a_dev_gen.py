from typing import List, Dict

from cache import load_from_pickle
from dataset_specific.msmarco.common import QueryID
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from tlm.data_gen.adhoc_datagen import FirstAndTitle
from tlm.data_gen.msmarco_doc_gen.gen_worker_qd import QueryDocEntityConcatPointwisePredictionGen
from tlm.data_gen.msmarco_doc_gen.mmd_worker import MMDWorker
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceTitleBodyCorpusWise
from tlm.qtype.content_functional_parsing.derived_query_set import load_derived_query_set_a_small
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import get_qid_to_content_tokens
from tlm.qtype.qde_resource import QDEResource


def run_for_prediction_split(split):
    print('qe_de_query_a_dev_gen.py')
    query_set = load_derived_query_set_a_small(split)
    candidate_docs_d = load_from_pickle("MMD_candidate_docs_d_{}".format(split))
    resource_source = ProcessedResourceTitleBodyCorpusWise(candidate_docs_d, split)
    resource = QDEResource(resource_source, query_set)
    q_max_seq_length = 128
    max_seq_length = 512
    basic_encoder = FirstAndTitle(max_seq_length)
    qid_to_entity_tokens: Dict[QueryID, List[str]] = get_qid_to_content_tokens(split)
    qid_to_entity_tokens = query_set.extend_query_id_based_dict(qid_to_entity_tokens)
    generator = QueryDocEntityConcatPointwisePredictionGen(resource, basic_encoder.encode, max_seq_length,
                                                           q_max_seq_length,
                                                           qid_to_entity_tokens
                                                           )

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunnerS(job_man_dir, len(resource.query_group), "MMD_{}_qe_de_a".format(split), factory)
    runner.start()


if __name__ == "__main__":
    # split = "dev"
    run_for_prediction_split("dev")
    run_for_prediction_split("test")