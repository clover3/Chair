import numpy as np
import tensorflow as tf

init_w = np.array([1.0])
learning_rate = 1e-2
var_w = tf.Variable(init_w, trainable=True)
optimizer = tf.keras.optimizers.Adam(learning_rate)
grad_array = np.array([1.0])
optimizer.apply_gradients([(grad_array, var_w)])
cur_w = var_w.value().numpy()
print(cur_w)

