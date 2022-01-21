import os
from collections import defaultdict

from cache import load_list_from_jsonl
from cpath import output_path
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import ComplementSearchOutput
from typing import List, Iterable, Callable, Dict, Tuple, Set


def load_complements() -> List[ComplementSearchOutput]:
    run_names = ["span_iter", "q_vector", "original_query"]
    cs_output_list_list = []
    for run_name in run_names:
        cs_output_list = load_complement_from_run(run_name)
        cs_output_list_list.append(cs_output_list)
        non_empty_len = len([cs for cs in cs_output_list if cs.complement_list])
        print("Run={} Complements found from {} of {}".format(run_name, non_empty_len, len(cs_output_list)))
    cs_output_list = join_complement_search_outputs(cs_output_list_list)
    return cs_output_list


def load_complement_from_run(run_name):
    save_dir = os.path.join(output_path, "qtype", "comp_search")
    save_path = os.path.join(save_dir, "{}.jsonl".format(run_name))
    cs_output_list: List[ComplementSearchOutput] = load_list_from_jsonl(save_path, ComplementSearchOutput.from_json)
    return cs_output_list


def join_complement_search_outputs(ll: List[List[ComplementSearchOutput]]) -> List[ComplementSearchOutput]:
    complements_d = defaultdict(list)
    for l in ll:
        for e in l:
            sig = e.problem_id, e.target_seg_idx
            complements_d[sig].extend(e.complement_list)

    outputs: List[ComplementSearchOutput] = []
    for sig, complement_list in complements_d.items():
        problem_id, target_seg_idx = sig
        cso = ComplementSearchOutput(problem_id, target_seg_idx, complement_list)
        outputs.append(cso)
    return outputs