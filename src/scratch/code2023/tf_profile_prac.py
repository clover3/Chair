from cpath import output_path
from misc_lib import path_join

import tensorflow as tf
import numpy as np

# Generate synthetic data for demonstration
num_samples = 10000
input_dim = 5  # Number of input features
output_dim = 1  # Number of output features (for regression)

# Random data and labels
data = np.random.random((num_samples, input_dim)).astype(np.float32)
labels = np.random.random((num_samples, output_dim)).astype(np.float32)

# Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(output_dim)
])

# Prepare the dataset
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

# Define optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_function = tf.keras.losses.MeanSquaredError()

# Training step function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_function(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop function
def train_fn():
    for inputs, labels in dataset:
        loss = train_step(inputs, labels)
        # Add any logging or metric tracking here

# Profiling the training loop
num_steps = 3  # Number of steps for profiling

save_dir = path_join(output_path, 'logdir')
tf.profiler.experimental.start(save_dir)
for step in range(num_steps):
    with tf.profiler.experimental.Trace("Train", step_num=step):
        train_fn()
tf.profiler.experimental.stop()
