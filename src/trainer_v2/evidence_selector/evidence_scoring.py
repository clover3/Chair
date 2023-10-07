import numpy as np

from trainer_v2.chair_logging import c_log


def cross_entropy(pred_prob, gold_prob) -> float:
    eps = 0.0000001
    v = - pred_prob * np.log(gold_prob) - (1-pred_prob) * np.log(1 - gold_prob + eps)
    return float(np.sum(v))


def mean_absolute_error(pred, gold) -> float:
    return float(np.sum(np.abs(pred - gold)))


def length_loss(num_used, max_num_tokens):
    return num_used / max_num_tokens


# Higher the better
def evidence_score(base_pred, rep_pred, num_used, n_p_tokens) -> float:
    err = cross_entropy(np.array(base_pred), np.array(rep_pred))  # [0, inf]
    l_loss = length_loss(num_used, n_p_tokens)
    tolerance = 0.05
    err_cap = 10
    err = min(err, err_cap)  # [0, 5]
    err = max(tolerance, err)
    combined_score = (err_cap-err) - tolerance * l_loss
    msg = f"CE={err:.2f} Usage={num_used}/{n_p_tokens} combined={combined_score}"
    c_log.info(msg)
    return combined_score


