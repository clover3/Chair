import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Input, LSTM, Dense, Embedding


def make_model(batch_size=None):
    maxlen = 10
    source = Input(shape=(maxlen,), batch_size=batch_size,
                   dtype=tf.int32, name='Input')
    embedding = Embedding(input_dim=128,
                          output_dim=128, name='Embedding')(source)
    lstm = LSTM(32, name='LSTM')(embedding)
    predicted_var = Dense(1, activation='sigmoid', name='Output')(lstm)
    model = tf.keras.Model(inputs=[source], outputs=[predicted_var])
    model.compile(
        optimizer=tf.compat.v1.train.RMSPropOptimizer(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['acc'])
    return model


training_model = make_model(batch_size=128)

# This address identifies the TPU we'll use when configuring TensorFlow.

tpu_model = tf.contrib.tpu.keras_to_tpu_model(
    training_model,
    strategy=tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(
                "v2-tf2")))


x_train = np.zeros([8, 10, 128])
y_train = np.zeros([8])

history = tpu_model.fit(x_train, y_train,
                        epochs=20,
                        batch_size=8,
                        validation_split=0.2)
tpu_model.save_weights('./tpu_model.h5', overwrite=True)
