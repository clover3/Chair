import itertools
from bert_api.task_clients.nli_interface.nli_predictors import get_nli_cache_client

from alignment import RelatedEvalAnswer, RelatedEvalInstance
from alignment.data_structure.eval_data_structure import join_a_p
from alignment.nli_align_path_helper import get_tfrecord_path, load_mnli_rei_problem
from alignment.perturbation_feature.segments_to_features_row_wise import make_tf_feature, \
    get_features
from alignment.perturbation_feature.train_configs import get_pert_train_data_shape
from alignment.related.related_answer_data_path_helper import load_related_eval_answer
from misc_lib import TimeEstimator, tprint
from tf_util.record_writer_wrap import RecordWriterWrap
from typing import List, Iterable, Callable, Dict, Tuple, Set


def main():
    dataset_name = "train"
    scorer_name = "lexical_v1"
    save_name = "train_2k_row"
    max_item = 2000

    tprint('Loading nli client')
    nli_client = get_nli_cache_client("localhost")
    tprint('Loading lexical alignment')
    answer_list: List[RelatedEvalAnswer] = load_related_eval_answer(dataset_name, scorer_name)
    answer_list = answer_list[:max_item]
    tprint('Loading mnli rei problems')
    problem_list: List[RelatedEvalInstance] = load_mnli_rei_problem(dataset_name)
    save_path = get_tfrecord_path(save_name)
    shape = get_pert_train_data_shape()
    ticker = TimeEstimator(max_item)
    writer = RecordWriterWrap(save_path)
    for answer, problem in join_a_p(answer_list, problem_list):
        for x, y in get_features(answer, problem, nli_client):
            feature = make_tf_feature(x, y, shape)
            writer.write_feature(feature)
        ticker.tick()

    print("{} records written".format(writer.total_written))


if __name__ == "__main__":
    main()