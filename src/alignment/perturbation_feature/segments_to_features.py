from typing import List, Tuple, Iterable

import numpy as np

from alignment import RelatedEvalAnswer, RelatedEvalInstance
from alignment.data_structure.eval_data_structure import join_a_p
from alignment.extract_feature import pairwise_feature
from alignment.nli_align_path_helper import load_mnli_rei_problem
from alignment.related.related_answer_data_path_helper import load_related_eval_answer
from bert_api import SegmentedInstance
from bert_api.task_clients.nli_interface.nli_interface import NLIInput


def build_x(nli_client,
            si: SegmentedInstance) -> np.array:
    # X_ij = [n, 3] (Probabilities for each perturbation)
    # X = [seg1_len, seg2_len, n, 3]
    x_building = []
    for seg1_idx in si.text1.enum_seg_idx():
        x_part = []
        for seg2_idx in si.text2.enum_seg_idx():
            si_list: List[SegmentedInstance] = pairwise_feature(si.text1, si.text2, seg1_idx, seg2_idx)
            todo = [NLIInput(si.text2, si.text1) for si in si_list]
            logits_list = nli_client.predict(todo)
            logits_list_np = np.array(logits_list)
            x_part.append(logits_list_np)
        x_part_np = np.stack(x_part, 0)
        x_building.append(x_part_np)
    return np.stack(x_building, 0)


def build_x_y(dataset_name, nli_client, scorer_name) -> Iterable[Tuple[np.array, np.array]]:
    answer_list: List[RelatedEvalAnswer] = load_related_eval_answer(dataset_name, scorer_name)
    problem_list: List[RelatedEvalInstance] = load_mnli_rei_problem(dataset_name)
    for answer, problem in join_a_p(answer_list, problem_list):
        alignment = answer.contribution.table
        seg_inst = problem.seg_instance
        x: np.array = build_x(nli_client, seg_inst)
        y = np.array(alignment)
        yield x, y