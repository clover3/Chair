from collections import OrderedDict
from typing import List, Tuple, Iterable

import numpy as np

from alignment import RelatedEvalAnswer, RelatedEvalInstance
from alignment.data_structure.eval_data_structure import join_a_p
from alignment.extract_feature import pairwise_feature
from alignment.nli_align_path_helper import load_mnli_rei_problem
from alignment.related.related_answer_data_path_helper import load_related_eval_answer
from bert_api import SegmentedInstance
from bert_api.task_clients.nli_interface.nli_interface import NLIInput
from data_generator.create_feature import create_float_feature
from list_lib import list_equal


def build_x(nli_client, si: SegmentedInstance, seg1_idx) -> np.array:
    # X_ij = [n, 3] (Probabilities for each perturbation)
    # X = [seg2_len, n, 3]
    x_part = []
    for seg2_idx in si.text2.enum_seg_idx():
        si_list: List[SegmentedInstance] = pairwise_feature(si.text1, si.text2, seg1_idx, seg2_idx)
        todo = [NLIInput(si.text2, si.text1) for si in si_list]
        logits_list = nli_client.predict(todo)
        logits_list_np = np.array(logits_list)
        x_part.append(logits_list_np)
    return np.stack(x_part, 0)



def length_match_check(answer, alignment, problem):
    seg_inst = problem.seg_instance
    seg1_len = len(alignment)
    l1_p = seg_inst.text1.get_seg_len()
    l2_p = seg_inst.text2.get_seg_len()
    l1_a = len(alignment)
    l2_a = len(alignment[0])
    if not l1_p == l1_a:
        print("{} / {}".format(answer.problem_id, problem.problem_id))
        print(f"({l1_p}, {l2_p}) ({l1_a}, {l2_a})")
        print("{} != {}".format(l1_p, l1_a))
    if not l2_p == l2_a:
        print("{} != {}".format(len(alignment[0]), seg_inst.text2.get_seg_len()))


def get_features(answer, problem, nli_client):
    length_match_check(answer, answer.contribution.table, problem)
    alignment = answer.contribution.table
    seg_inst = problem.seg_instance

    for seg1_idx in seg_inst.text1.enum_seg_idx():
        scores_row = alignment[seg1_idx]
        x: np.array = build_x(nli_client, seg_inst, seg1_idx)
        y = np.array(scores_row)
        yield x, y


def make_tf_feature(x, y, shape) -> OrderedDict:
    n_max_seg = shape[1]
    x_slice = x[:n_max_seg, :, :]
    y_slice = y[:n_max_seg]

    n_pad2 = shape[1] - x_slice.shape[0]

    x_padded = np.pad(x_slice, [(0, n_pad2), (0, 0), (0, 0)])
    y_padded = np.pad(y_slice, [(0, n_pad2)])

    if not list_equal(list(x_padded.shape), shape[1:]):
        print("{} != {}".format(x_padded.shape, shape[1:]))
    if not list_equal(list(y_padded.shape), shape[1:2]):
        print("{} != {}".format(y_padded.shape, shape[1:2]))

    def encode_np_array(np_array):
        np_flat = np.reshape(np_array, [-1])
        return create_float_feature(np_flat)

    features = OrderedDict()
    features['x'] = encode_np_array(x_padded)
    features['y'] = encode_np_array(y_padded)
    return features