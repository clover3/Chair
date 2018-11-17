

from data_generator.stance.stance_detection import stance_label
import numpy as np

def stance_f1(predictions, gold):
    idx_none = stance_label.index("NONE")
    idx_favor = stance_label.index("FAVOR")
    idx_against= stance_label.index("AGAINST")

    def f1_per_flag(flag_idx):
        arr_gold_pos = np.equal(gold, flag_idx)
        arr_pred_pos = np.equal(predictions, flag_idx)
        arr_true_pos = np.logical_and(arr_gold_pos, arr_pred_pos)

        n_true_pos = np.count_nonzero(arr_true_pos)
        n_pred_pos = np.count_nonzero(arr_pred_pos)
        n_gold = np.count_nonzero(arr_gold_pos)

        prec = n_true_pos / n_pred_pos if n_true_pos > 0 else 0
        recall = n_true_pos / n_gold if n_gold > 0 else 0

        if (prec + recall) == 0:
            return 0
        else:
            return 2*prec*recall / (prec + recall)

    f1_favor = f1_per_flag(idx_favor)
    f1_against = f1_per_flag(idx_against)

    return (f1_favor + f1_against) / 2