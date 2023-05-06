import tensorflow as tf
target_q_term_mask = tf.constant([[0, 1, 0, 0,]])
target_d_term_mask = tf.constant([[0, 0, 0, 1,]])
a = tf.expand_dims(target_q_term_mask, axis=2)
b = tf.expand_dims(target_d_term_mask, axis=1)
qd_target_mask = a * b

print(qd_target_mask)