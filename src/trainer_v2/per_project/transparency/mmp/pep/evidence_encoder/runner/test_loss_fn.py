import tensorflow as tf

from trainer_v2.per_project.transparency.mmp.pep.evidence_encoder.ee_train_model import loss_per_qd


def wrap_tensor_compute(is_valid_mask, label_arr, pred_arr, loss_fn):
    pred_arr = tf.constant(pred_arr)
    label_arr = tf.constant(label_arr)
    is_valid_mask = tf.constant(is_valid_mask)
    loss = loss_fn(pred_arr, label_arr, is_valid_mask)
    return loss


def loss_per_qd_demo(pred_score_arr, label_scores_stack, is_valid_mask):
    bias = (1-is_valid_mask) * -1000000.0
    pred_score_arr = pred_score_arr + bias  # [B, n_qt, n_dt]
    print("pred_score_arr", pred_score_arr.numpy())
    pred_probs = tf.nn.softmax(pred_score_arr, axis=2)
    print("pred_probs", pred_probs.numpy())

    label_scores_stack = label_scores_stack + bias
    print("label_scores_stack", label_scores_stack.numpy())
    label_probs = tf.nn.softmax(label_scores_stack, axis=2)
    print("label_probs", label_probs.numpy())

    print("pred_probs * label_probs", (pred_probs * label_probs).numpy())

    per_item_loss = -tf.reduce_sum(tf.reduce_sum(pred_probs * label_probs, axis=2), axis=1)
    return per_item_loss


def main():
    pred_arr = [[
        [0.1, 9., 0.2],
    ]]
    label_arr = [[
        [0.1, 0.4, 0.2]
    ]]
    is_valid_mask = [[
        [1., 1., 0.]
    ]]
    loss_fn = loss_per_qd_demo
    loss = wrap_tensor_compute(is_valid_mask, label_arr, pred_arr, loss_fn)
    print(loss.numpy().tolist())
    pred_arr = [[
        [0.1, 9., 0.2],
    ]]
    label_arr = [[
        [0.1, 0.2, 0.4]
    ]]
    loss_fn = loss_per_qd
    loss = wrap_tensor_compute(is_valid_mask, label_arr, pred_arr, loss_fn)
    print(loss.numpy().tolist())



if __name__ == "__main__":
    main()