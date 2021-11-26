from typing import Dict

from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.common import load_candidate_doc_list_10
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from tlm.data_gen.msmarco_doc_gen.mmd_worker import MMDWorker
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceTitleBodyTokensListTrain
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import get_qid_to_content_tokens, get_qid_to_qtype_id
from tlm.qtype.gen_qtype.qtype_prediction_gen import QTypeIDPredictionGen

if __name__ == "__main__":
    split = "train"
    resource = ProcessedResourceTitleBodyTokensListTrain(split, load_candidate_doc_list_10)
    max_seq_length = 128
    tokenizer = get_tokenizer()
    qid_to_entity_tokens = get_qid_to_content_tokens(split)
    qid_to_qtype_id: Dict[str, int] = get_qid_to_qtype_id(split)

    generator = QTypeIDPredictionGen(resource, max_seq_length,
                                        qid_to_entity_tokens, qid_to_qtype_id)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunnerS(job_man_dir, len(resource.query_group)-1, "MMD_{}_qtype_prediction".format(split), factory)
    runner.start()
