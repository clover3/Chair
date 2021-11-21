import os
from typing import Dict

from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.common import load_candidate_doc_list_10
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from tlm.data_gen.msmarco_doc_gen.gen_qtype.distill_gen_worker import DistillGenWorker
from tlm.data_gen.msmarco_doc_gen.gen_worker_qd import FixedQTypeIDDistillGen
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceTitleBodyTokensListTrain
from tlm.qtype.qid_to_content_tokens import get_qid_to_content_tokens, get_qid_to_qtype_id

if __name__ == "__main__":
    split = "train"
    q_max_seq_length = 128
    max_seq_length = 512
    tokenizer = get_tokenizer()
    resource = ProcessedResourceTitleBodyTokensListTrain(split, load_candidate_doc_list_10)
    qid_to_entity_tokens = get_qid_to_content_tokens(split)
    qid_to_qtype_id: Dict[str, int] = get_qid_to_qtype_id(split)
    resource_path_format = os.path.join(job_man_dir, "MMD_2M_10doc_parse", "{}")
    generator = FixedQTypeIDDistillGen(resource,
                                       max_seq_length,
                                       qid_to_entity_tokens,
                                       qid_to_qtype_id
                                       )

    def factory(out_dir):
        return DistillGenWorker(generator, resource.query_group, resource_path_format, out_dir)

    runner = JobRunnerS(job_man_dir, len(resource.query_group)-1, "MMD_{}_fixed_qtype_de_distill_10doc".format(split), factory)
    runner.start()
