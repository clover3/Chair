from alignment.data_structure.eval_data_structure import RelatedBinaryAnswer
from alignment.perturbation_feature.segments_to_features_row_wise import length_match_check, build_x
import numpy as np
from typing import List, Iterable, Callable, Dict, Tuple, Set


def get_features_skip_empty(answer: RelatedBinaryAnswer, problem, nli_client):
    alignment = answer.score_table
    seg_inst = problem.seg_instance
    length_match_check(answer, answer.score_table, problem)

    for seg1_idx in seg_inst.text1.enum_seg_idx():
        scores_row: List[int] = alignment[seg1_idx]
        if all([v == 0 for v in scores_row]):
            pass
        else:
            x: np.array = build_x(nli_client, seg_inst, seg1_idx)
            y = np.array(scores_row)
            yield x, y
