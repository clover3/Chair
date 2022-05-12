import numpy as np
import tensorflow as tf

from alignment import MatrixScorerIF
from alignment.data_structure.matrix_scorer_if import ContributionSummary
from alignment.perturbation_feature.pert_model_1d import binary_hinge_loss, precision_at_1
from alignment.perturbation_feature.segments_to_features_row_wise import build_x
from alignment.perturbation_feature.train_configs import get_pert_train_data_shape_1d
from alignment.related.related_answer_data_path_helper import get_model_save_path
from bert_api import SegmentedInstance
from bert_api.task_clients.nli_interface.nli_predictors import get_nli_cache_client


def load_pert_pred_model(model_name, eval_data=None):
    custom_objects = {'binary_hinge_loss': binary_hinge_loss, 'precision_at_1': precision_at_1}
    new_model = tf.keras.models.load_model(get_model_save_path(model_name),
                                           custom_objects=custom_objects)
    if eval_data is not None:
        new_model.evaluate(eval_data, batch_size=8)
    return new_model


class LearnedPerturbationScorer(MatrixScorerIF):
    def __init__(self, new_model):
        self.new_model = new_model
        self.nli_client = get_nli_cache_client("localhost")
        self.shape = get_pert_train_data_shape_1d()

    def _pad_x(self, x):
        n_max_seg = self.shape[0]
        x_slice = x[:n_max_seg, :, :]
        n_pad2 = self.shape[0] - x_slice.shape[0]
        x_padded = np.pad(x_slice, [(0, n_pad2), (0, 0), (0, 0)])
        return x_padded

    def eval_contribution(self, inst: SegmentedInstance) -> ContributionSummary:
        x_batch_items = []
        for seg1_idx in inst.text1.enum_seg_idx():
            x: np.array = build_x(self.nli_client, inst, seg1_idx)
            pad_x = self._pad_x(x)
            x_flat = np.reshape(pad_x, [-1, self.shape[1] * self.shape[2]])
            x_batch_items.append(x_flat)

        X = np.stack(x_batch_items, axis=0)
        y = self.new_model.predict(X)  # [num_segs, 256, 1]
        y = y[:, :inst.text2.get_seg_len(), 0]
        y_list = y.tolist()
        return ContributionSummary(y_list)
