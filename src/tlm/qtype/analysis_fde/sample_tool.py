import os

from cache import load_pickle_from
from cpath import output_path
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import QueryInfo


def enum_sample_entries(run_name, job_id):
    save_path = os.path.join(output_path, "qtype", run_name + '_sample', str(job_id))
    qtype_entries, query_info_dict = load_pickle_from(save_path)

    for e in qtype_entries:
        info: QueryInfo = query_info_dict[e.qid]
        func_span_rep = info.get_func_span_rep()
        yield e, info, func_span_rep