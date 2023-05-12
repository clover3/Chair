import tensorflow as tf
import numpy as np


def main():
    B = 2
    M = 16
    y_pred_score = tf.ones([B, 12, 1])
    print("y_pred_score", y_pred_score)

    label = tf.constant([0, 1])  # [B ]
    print('label', label)

    valid_mask_np = np.zeros([B, M], int)
    valid_mask_np[0, 1] = 1
    valid_mask_np[1, 1] = 1

    valid_mask = tf.constant(valid_mask_np)  # [B, M]
    print("valid_mask", valid_mask.numpy())
    valid_r = tf.reduce_max(valid_mask, axis=1, keepdims=True)  # [B, 1]
    print("valid_r", valid_r.numpy())

    sample_weight = tf.cast(valid_r, tf.float32)  # [B, 1]
    y_pred = y_pred_score > 0  # [B, 12, 1]
    label_b = tf.cast(label, tf.bool)  # [B, ]
    print("label_b", label_b.numpy())
    label_b_ex = tf.expand_dims(tf.expand_dims(label_b, axis=1), axis=1)  # [B, 1, 1]
    print("tf.equal(label_b_ex, y_pred)", tf.equal(label_b_ex, y_pred).numpy())
    is_correct = tf.cast(tf.equal(label_b_ex, y_pred), tf.int32)  # [B, 12, 1]
    n_valid_correct = tf.reduce_sum(is_correct * tf.expand_dims(valid_r, axis=1))  #
    correct_added = tf.cast(n_valid_correct, tf.float32)
    n_valid = tf.reduce_sum(tf.cast(sample_weight, tf.float32))
    print("correct_added", correct_added)
    print("n_valid", n_valid)


if __name__ == "__main__":
    main()