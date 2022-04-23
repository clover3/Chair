import itertools

from alignment.data_structure.matrix_scorer_if import ContributionSummary
from alignment.matrix_scorers.related_scoring_common import run_scoring
from alignment.perturbation_feature.pert_model_1d import binary_hinge_loss
from alignment.perturbation_feature.pert_model_2d import precision_at_1
from alignment.perturbation_feature.segments_to_features_row_wise import make_tf_feature, get_features, build_x
from bert_api import SegmentedInstance
from bert_api.task_clients.nli_interface.nli_interface import get_nli_cache_client

from alignment import RelatedEvalAnswer, RelatedEvalInstance, MatrixScorerIF
from alignment.nli_align_path_helper import get_tfrecord_path, load_mnli_rei_problem, save_related_eval_answer
from alignment.perturbation_feature.train_configs import get_pert_train_data_shape, get_pert_train_data_shape_1d
from alignment.related.related_answer_data_path_helper import load_related_eval_answer, load_binary_related_eval_answer, \
    get_model_save_path
from misc_lib import TimeEstimator, tprint
from typing import List, Iterable, Callable, Dict, Tuple, Set
import numpy as np
import tensorflow as tf


class LearnedPerturbationScorer(MatrixScorerIF):
    def __init__(self, model_nave):
        custom_objects = { 'binary_hinge_loss': binary_hinge_loss, 'precision_at_1': precision_at_1}
        new_model = tf.keras.models.load_model(get_model_save_path(model_nave),
                                               custom_objects=custom_objects)
        self.new_model = new_model
        self.nli_client = get_nli_cache_client("localhost")
        self.n_feature = 9
        self.n_classes = 3
        self.shape = get_pert_train_data_shape_1d()

    def pad_x(self, x):
        n_max_seg = self.shape[0]
        x_slice = x[:n_max_seg, :, :]
        n_pad2 = self.shape[0] - x_slice.shape[0]
        x_padded = np.pad(x_slice, [(0, n_pad2), (0, 0), (0, 0)])
        return x_padded

    def eval_contribution(self, inst: SegmentedInstance) -> ContributionSummary:
        x_batch_items = []
        for seg1_idx in inst.text1.enum_seg_idx():
            x: np.array = build_x(self.nli_client, inst, seg1_idx)
            pad_x = self.pad_x(x)
            x_flat = np.reshape(pad_x, [-1, self.shape[1] * self.shape[2]])
            x_batch_items.append(x_flat)

        X = np.stack(x_batch_items, axis=0)
        y = self.new_model.predict(X) # [num_segs, 256, 1]
        y = y[:, :inst.text2.get_seg_len(), 0]
        y_list = y.tolist()
        return ContributionSummary(y_list)


def main():
    dataset_name = "train_head"
    model_name = 'train_v1_1_2K_linear_9'
    scorer_name = "pert_" + model_name
    scorer = LearnedPerturbationScorer(model_name)
    problem_list: List[RelatedEvalInstance] = load_mnli_rei_problem(dataset_name)
    problem_list = problem_list[:10]
    answers: List[RelatedEvalAnswer] = run_scoring(problem_list, scorer)
    save_related_eval_answer(answers, dataset_name, scorer_name)


if __name__ == "__main__":
    main()