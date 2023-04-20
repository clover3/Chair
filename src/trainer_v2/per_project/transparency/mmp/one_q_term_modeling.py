import numpy as np
import tensorflow as tf

from trainer_v2.custom_loop.dataset_factories import create_dataset_common
from trainer_v2.custom_loop.run_config2 import RunConfig2


class ScoringLayer2(tf.keras.layers.Layer):
    def __init__(self, voca_size, **kwargs):
        super(ScoringLayer2, self).__init__(**kwargs)
        self.k1 = 0.1
        self.b = 1.2
        self.avdl = 50
        self.voca_size = voca_size
        init_w = np.zeros([self.voca_size], np.float)
        self.w = tf.Variable(init_w, trainable=True, dtype=tf.float32)
        self.qtw = tf.Variable(np.ones([1], np.float), trainable=True, dtype=tf.float32)
        # self.qtw = tf.constant([1.0], dtype=tf.float32)

    def call(self, x):
        # [B, V]
        x = tf.cast(x, tf.float32)
        t = x * self.w
        d_tf = tf.reduce_sum(t, axis=1)
        d_tf = tf.maximum(d_tf, 0)
        denom = d_tf + d_tf * self.k1
        nom = d_tf + self.k1
        dtw = denom / (nom + 1e-8)
        score = self.qtw * dtw
        return score


class ScoringLayer3(tf.keras.layers.Layer):
    def __init__(self, voca_size, init_w, init_qtw, **kwargs):
        super(ScoringLayer3, self).__init__(**kwargs)
        self.k1 = 0.1
        self.b = 1.2
        self.avdl = 50
        self.voca_size = voca_size
        self.w = tf.Variable(init_w, trainable=True, dtype=tf.float32)
        self.qtw = tf.Variable(init_qtw, trainable=True, dtype=tf.float32)
        # self.qtw = tf.constant([1.0], dtype=tf.float32)

    def call(self, x):
        # [B, V]
        x = tf.cast(x, tf.float32)
        t = x * self.w
        d_tf = tf.reduce_sum(t, axis=1)
        d_tf = tf.maximum(d_tf, 0)
        denom = d_tf + d_tf * self.k1
        nom = d_tf + self.k1
        dtw = denom / (nom + 1e-8)
        score = self.qtw * dtw
        return score


class ScoringLayer4(tf.keras.layers.Layer):
    def __init__(self, init_w, init_qtw, **kwargs):
        super(ScoringLayer4, self).__init__(**kwargs)
        k1_init = 0.1
        self.k1 = tf.Variable(k1_init, trainable=True, dtype=tf.float32)
        self.w = tf.Variable(init_w, trainable=True, dtype=tf.float32)
        self.qtw = tf.Variable(init_qtw, trainable=True, dtype=tf.float32)
        # self.qtw = tf.constant([1.0], dtype=tf.float32)

    def call(self, x):
        # [B, V]
        x = tf.cast(x, tf.float32)
        t = x * self.w
        d_tf = tf.reduce_sum(t, axis=1)
        d_tf = tf.maximum(d_tf, 0)
        denom = d_tf + d_tf * self.k1
        nom = d_tf + self.k1
        dtw = denom / (nom + 1e-8)
        score = self.qtw * dtw
        return score


def get_dataset(
        file_path,
        voca_size,
        run_config: RunConfig2,
    ) -> tf.data.Dataset:
    max_seq_length = 32

    def decode_record(record):
        name_to_features = {
        }
        for i in range(2):
            def fixed_len_feature():
                return tf.io.FixedLenFeature([max_seq_length], tf.int64)
            name_to_features[f'feature_ids{i+1}'] = fixed_len_feature()
            name_to_features[f'feature_values{i+1}'] = fixed_len_feature()
            name_to_features[f'score{i+1}'] = tf.io.FixedLenFeature([1], tf.float32)

        record = tf.io.parse_single_example(record, name_to_features)
        return reform_example(record)

    def reform_example(record):

        for i in range(2):
            feature_ids = record[f'feature_ids{i+1}']
            feature_values = record[f'feature_values{i+1}']
            x = tf.scatter_nd(tf.expand_dims(feature_ids, 1),
                              feature_values, [voca_size])
            record[f'x{i+1}'] = x
        return record, [tf.constant(1)]

    return create_dataset_common(decode_record,
                                 run_config,
                                 file_path,
                                 False)


def get_model(voca_size, run_config):
    is_training = run_config.is_training()
    score1 = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name="score1")
    x1 = tf.keras.layers.Input(shape=(voca_size,), dtype=tf.int32, name="x1")

    score2 = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name="score2")
    x2 = tf.keras.layers.Input(shape=(voca_size,), dtype=tf.int32, name="x2")

    network = ScoringLayer2(voca_size)
    logits1 = network(x1)
    logits2 = network(x2)
    learning_rate = run_config.train_config.learning_rate
    score1_f = score1 + logits1
    score2_f = score2 + logits2
    # inputs = [feature_ids1, feature_values1, score1, feature_ids2, feature_values2, score2]
    inputs = [x1, score1, x2, score2]
    pred = score1_f - score2_f
    print("pred", pred)
    new_model = tf.keras.models.Model(inputs=inputs, outputs=pred)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    new_model.add_loss(lambda : tf.reduce_sum(tf.abs(network.w)) * 0.01)
    new_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Hinge(),
        steps_per_execution=run_config.common_run_config.steps_per_execution,
    )
    return new_model


def get_model2(voca_size, when_id, run_config):
    is_training = run_config.is_training()
    score1 = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name="score1")
    x1 = tf.keras.layers.Input(shape=(voca_size,), dtype=tf.int32, name="x1")

    score2 = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name="score2")
    x2 = tf.keras.layers.Input(shape=(voca_size,), dtype=tf.int32, name="x2")

    init_w = np.zeros([voca_size])
    init_w[when_id] = 1
    init_qtw = 3
    network = ScoringLayer4(init_w, init_qtw)
    logits1 = network(x1)
    logits2 = network(x2)
    learning_rate = run_config.train_config.learning_rate
    score1_f = score1 + logits1
    score2_f = score2 + logits2
    # inputs = [feature_ids1, feature_values1, score1, feature_ids2, feature_values2, score2]
    inputs = [x1, score1, x2, score2]
    pred = score1_f - score2_f
    print("pred", pred)
    new_model = tf.keras.models.Model(inputs=inputs, outputs=pred)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    new_model.add_loss(lambda : tf.reduce_sum(tf.abs(network.w)) * 0.01)
    new_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Hinge(),
        steps_per_execution=run_config.common_run_config.steps_per_execution,
    )
    return new_model