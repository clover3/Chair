import itertools

import numpy as np

from alignment.extract_feature import pairwise_feature4
from alignment.perturbation_feature.segments_to_features_row_wise import make_tf_feature, \
    length_match_check
from bert_api import SegmentedInstance
from bert_api.task_clients.nli_interface.nli_interface import get_nli_cache_client, NLIInput

from alignment import RelatedEvalAnswer, RelatedEvalInstance
from alignment.data_structure.eval_data_structure import join_a_p, RelatedBinaryAnswer
from alignment.nli_align_path_helper import get_tfrecord_path, load_mnli_rei_problem
from alignment.perturbation_feature.train_configs import get_pert_train_data_shape, get_pert_train4_data_shape
from alignment.related.related_answer_data_path_helper import load_related_eval_answer, load_binary_related_eval_answer
from misc_lib import TimeEstimator, tprint
from tf_util.record_writer_wrap import RecordWriterWrap
from typing import List, Iterable, Callable, Dict, Tuple, Set


def build_x(nli_client, si: SegmentedInstance, seg1_idx) -> np.array:
    # X_ij = [n, 3] (Probabilities for each perturbation)
    # X = [seg2_len, n, 3]
    x_part = []
    for seg2_idx in si.text2.enum_seg_idx():
        si_list: List[SegmentedInstance] = pairwise_feature4(si.text1, si.text2, seg1_idx, seg2_idx)
        todo = [NLIInput(si.text2, si.text1) for si in si_list]
        logits_list = nli_client.predict(todo)
        logits_list_np = np.array(logits_list)
        x_part.append(logits_list_np)
    return np.stack(x_part, 0)


def get_features(answer: RelatedBinaryAnswer, problem, nli_client):
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


def main():
    dataset_name = "train"
    scorer_name = "lexical_v1_1"
    size_k= 5
    save_name = f"{dataset_name}_4_{size_k}K"
    max_item = size_k * 1000
    tprint("Save name: " + save_name)

    tprint('Loading nli client')
    nli_client = get_nli_cache_client("localhost")
    tprint('Loading lexical alignment')
    answer_list: List[RelatedBinaryAnswer] = load_binary_related_eval_answer(dataset_name, scorer_name)
    answer_list = answer_list[:max_item]
    tprint('Loading mnli rei problems')
    problem_list: List[RelatedEvalInstance] = load_mnli_rei_problem(dataset_name)
    save_path = get_tfrecord_path(save_name)
    shape = get_pert_train4_data_shape()
    ticker = TimeEstimator(len(answer_list))

    writer = RecordWriterWrap(save_path)
    for answer, problem in join_a_p(answer_list, problem_list):
        for x, y in get_features(answer, problem, nli_client):
            feature = make_tf_feature(x, y, shape)
            writer.write_feature(feature)
        ticker.tick()

    print("{} records written".format(writer.total_written))


if __name__ == "__main__":
    main()
