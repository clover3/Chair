
import tensorflow as tf
import numpy as np
probs = tf.constant([0.9, 0.1, 0.2])
print("probs", probs)

for beta in [1, 2, 4, 7, 10]:
    loss_weight = tf.nn.softmax(probs * beta)
    print('beta', beta)
    print("loss_weight", np.array(loss_weight))
