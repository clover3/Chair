import os
from typing import List, Dict

from data_generator.tokenizer_wo_tf import get_tokenizer
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from misc_lib import tprint
from tlm.data_gen.msmarco_doc_gen.gen_worker_qd import QueryDocEntityDistilGen
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceTitleBodyTokensListPredict
from tlm.qtype.content_functional_parsing.derived_query_set import load_derived_query_set_a
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import get_qid_to_content_tokens
from tlm.qtype.gen_qtype.distill_gen_worker import DistillGenWorker10Grouped
from tlm.qtype.qde_resource import QDEResource
from trec.types import QueryID

if __name__ == "__main__":
    split = "train"
    q_max_seq_length = 128
    max_seq_length = 512
    tokenizer = get_tokenizer()
    tprint("load_derived_query_set_a")
    query_set = load_derived_query_set_a(split)
    tprint("ProcessedResourceTitleBodyTokensListPredict")
    resource_source = ProcessedResourceTitleBodyTokensListPredict(split)
    tprint("QDEResource")
    resource = QDEResource(resource_source, query_set)

    qid_to_entity_tokens_raw: Dict[QueryID, List[str]] = get_qid_to_content_tokens(split)
    qid_to_entity_tokens = query_set.extend_query_id_based_dict(qid_to_entity_tokens_raw)
    resource_path_format = os.path.join(job_man_dir, "MMD_Z_query_a_parse", "{}")
    generator = QueryDocEntityDistilGen(resource,
                                        max_seq_length,
                                        q_max_seq_length,
                                        qid_to_entity_tokens)

    def factory(out_dir):
        return DistillGenWorker10Grouped(generator, resource.query_group, resource_path_format, out_dir)

    n_jobs = int(len(resource.query_group) / 10) + 1
    runner = JobRunnerS(job_man_dir, n_jobs, "MMD_{}_qe_de_distill_prob".format(split), factory)
    runner.start()
