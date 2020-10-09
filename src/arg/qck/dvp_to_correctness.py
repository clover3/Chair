from typing import List, Dict, Tuple

from arg.qck.doc_value_calculator import DocValueParts
from list_lib import lmap
from misc_lib import group_by


def get_dvp_value(dvp: DocValueParts):
    return dvp.value


def get_dvp_qid(dvp: DocValueParts):
    return dvp.query.query_id


def get_doc_id_idx(dvp: DocValueParts):
    return dvp.kdp.doc_id, dvp.kdp.passage_idx


def dvp_to_correctness(dvp_list: List[DocValueParts],
                       run_config: Dict) -> Dict[Tuple[str, Tuple[str, int]], bool]:

    def is_good(values: List[float]) -> bool:
        #  value = abs(error_new) - abs(error_baseline)
        return sum(values) > 0

    grouped = group_by(dvp_list, get_dvp_qid)

    is_good_dict: Dict[Tuple[str, Tuple[str, int]], bool] = {}
    for qid, entries in grouped.items():
        g2 = group_by(entries, get_doc_id_idx)

        for doc_sig, entries2 in g2.items():
            values = lmap(get_dvp_value, entries2)
            is_good_dict[qid, doc_sig] = is_good(values)
    return is_good_dict