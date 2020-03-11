import datetime
import logging
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

from cpath import model_path
from icd.code import fit_and_tokenize, input_fn, str_code_id, str_desc_tokens
from icd.common import lmap, load_description
from icd.common_args import parser
from models.transformer.bert_common import create_initializer
from models.transformer.optimization import create_optimizer


def build_model(code_ids, token_ids, dim, max_seq, n_input_voca, n_output_voca):
    #W2 = tf.keras.layers.Dense(n_output_voca, use_bias=False)
    #W2 = K.random_normal_variable(shape=(n_output_voca, dim), mean=0, scale=1)
    np_val = np.reshape(np.random.normal(size=n_output_voca * dim),
                        [n_output_voca, dim])
    W2 = tf.constant(np_val, dtype=tf.float32)
    initializer_range = 0.1
    embedding_table = tf.compat.v1.get_variable(
            name="embedding",
            shape=[n_input_voca, dim],
            initializer=create_initializer(initializer_range),
            dtype=tf.float32
    )
    h0 = tf.nn.embedding_lookup(params=embedding_table, ids=code_ids)
    h = tf.reshape(h0, [-1, dim])
    h = tf.nn.l2_normalize(h, -1)
    W2 = tf.nn.l2_normalize(W2, -1)
    logits = tf.matmul(h, W2, transpose_b=True)
    logits_1 = tf.expand_dims(logits, 1)
    y = tf.one_hot(token_ids, depth=n_output_voca) #[batch, max_seq, n_output_voca]
    print("logits", logits.shape)
    print("y", y.shape)
    pos_val = logits_1 * y # [ batch, max_seq, voca]
    neg_val = logits - tf.reduce_sum(pos_val, axis=1) #[ batch, voca]
    t = tf.reduce_sum(pos_val, axis=2) # [batch, max_seq]
    correct_map = tf.expand_dims(t, 2) # [batch, max_seq, 1]
    print("correct_map", correct_map.shape)
    wrong_map = tf.expand_dims(neg_val, 1) # [batch, 1, voca]
    print(wrong_map.shape)
    t = wrong_map - correct_map + 1
    print("t", t.shape)
    loss = tf.reduce_mean(tf.math.maximum(t, 0), axis=-1)
    mask = tf.cast(tf.not_equal(token_ids, 0), tf.float32) # batch, seq_len
    print("mask", mask.shape)
    loss = mask * loss
    loss = tf.reduce_sum(loss, axis=1) # over the sequence
    loss = tf.reduce_mean(loss)
    print("loss", loss.shape)
    return loss


def build_model_fn(lr, dim, max_seq, n_input_voca, n_output_voca):
    def model_fn(features, labels, mode, params):
        code_id = features[str_code_id]
        token_ids = features[str_desc_tokens]

        #mode = tf.estimator.ModeKeys.TRAIN
        loss = build_model(code_id, token_ids, dim, max_seq, n_input_voca, n_output_voca)
        #logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=1)
        logging_hook = _LoggerHook(loss, 1)
        num_train_step = 1000
        train_op = create_optimizer(loss, lr, num_train_step, 0, False)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op
        )
    return model_fn


class _LoggerHook(tf.train.SessionRunHook):
    def __init__(self, loss, log_frequency):
        self.log_frequency = log_frequency
        self.loss = loss

    def begin(self):
        self._step = -1
        self._start_time = time.time()

    def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(self.loss)  # Asks for loss value.

    def after_run(self, run_context, run_values):
        if self._step % self.log_frequency == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time

            loss_value = run_values.results
            examples_per_sec = self.log_frequency * 16 / duration
            sec_per_batch = float(duration / self.log_frequency)

            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (datetime.datetime.now(), self._step, loss_value,
                                examples_per_sec, sec_per_batch))


def train_loop(args):
    model_dir = os.path.join(model_path, "w2v_model")
    data = load_description()
    n_input_voca = data[-1]['order_number']
    input2 = lmap(lambda x: x['short_desc'], data)
    dim = 1000
    max_seq = 1
    batch_size = int(args.batch_size)
    epochs = int(args.epochs)
    lr = float(args.lr)
    print("learning rate", lr)
    print("Batch_size", batch_size)
    enc_text, tokenizer = fit_and_tokenize(input2)
    token_config = tokenizer.get_config()
    n_output_voca = len(token_config['word_index'])

    random.shuffle(data)
    train_size = int(0.1 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    step_per_epoch = int(len(train_data) / batch_size)
    max_step = step_per_epoch * epochs
    config = tf.estimator.RunConfig().replace(keep_checkpoint_max=1,
                                              log_step_count_steps=10,
                                              save_checkpoints_steps=step_per_epoch)
    tf_logging = logging.getLogger('tensorflow')
    tf_logging.setLevel(logging.DEBUG)
    print("Building_estimator")
    estimator = tf.estimator.Estimator(
        model_dir=model_dir,
        model_fn=build_model_fn(lr, dim, max_seq, n_input_voca, n_output_voca),
        config=config
    )

    print("start training")
    # estimator.train(
    #     input_fn=lambda :input_fn(train_data, tokenizer, max_seq),
    #     steps=max_step
    # )
    estimator.predict(
        input_fn=lambda: input_fn(train_data, tokenizer, max_seq),
        steps=max_step
    )


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    train_loop(args)