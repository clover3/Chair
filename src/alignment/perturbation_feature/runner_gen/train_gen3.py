from typing import List

import numpy as np

from alignment import RelatedEvalInstance
from alignment.data_structure.eval_data_structure import join_a_p, RelatedBinaryAnswer
from alignment.nli_align_path_helper import get_tfrecord_path, load_mnli_rei_problem
from alignment.perturbation_feature.segments_to_features_row_wise import make_tf_feature, length_match_check, build_x
from alignment.perturbation_feature.train_configs import get_pert_train_data_shape
from alignment.related.related_answer_data_path_helper import load_binary_related_eval_answer
from bert_api.task_clients.nli_interface.nli_predictors import get_nli_cache_client
from misc_lib import TimeEstimator, tprint
from tf_util.record_writer_wrap import RecordWriterWrap


def main():
    dataset_name = "train"
    scorer_name = "lexical_v1_1"
    save_name = f"{dataset_name}_v1_1_100K"
    max_item = 100 * 1000
    tprint("Save name: " + save_name)

    tprint('Loading nli client')
    nli_client = get_nli_cache_client("localhost")
    tprint('Loading lexical alignment')
    answer_list: List[RelatedBinaryAnswer] = load_binary_related_eval_answer(dataset_name, scorer_name)
    answer_list = answer_list[:max_item]
    tprint('Loading mnli rei problems')
    problem_list: List[RelatedEvalInstance] = load_mnli_rei_problem(dataset_name)
    save_path = get_tfrecord_path(save_name)
    shape = get_pert_train_data_shape()
    ticker = TimeEstimator(len(answer_list))
    writer = RecordWriterWrap(save_path)
    for answer, problem in join_a_p(answer_list, problem_list):
        for x, y in get_features_skip_empty(answer, problem, nli_client):
            feature = make_tf_feature(x, y, shape)
            writer.write_feature(feature)
        ticker.tick()

    print("{} records written".format(writer.total_written))


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


if __name__ == "__main__":
    main()

