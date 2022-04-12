from typing import Dict, List

from list_lib import l_to_map, index_by_fn, dict_value_map
from alignment.data_structure.eval_data_structure import RelatedBinaryAnswer


def get_index_answer_dict(dataset, method_list, related_answer_load_fn):
    def load_answers(method):
        return related_answer_load_fn(dataset, method)
    answer_d_raw: Dict[str, List[RelatedBinaryAnswer]] = l_to_map(load_answers, method_list)

    def index_answer_list(l: List[RelatedBinaryAnswer]) -> Dict[str, RelatedBinaryAnswer]:
        return index_by_fn(lambda a: a.problem_id, l)

    answer_d: Dict[str, Dict[str, RelatedBinaryAnswer]] = dict_value_map(index_answer_list, answer_d_raw)
    return answer_d