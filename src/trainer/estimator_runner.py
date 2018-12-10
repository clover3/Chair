from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from task.transformer_est import Transformer, Classification
from models.transformer import hyperparams
from data_generator import shared_setting
from data_generator.stance import stance_detection
import pandas as pd
import path
from misc_lib import delete_if_exist

def get_model_dir(run_id, delete_exist = True):
    run_dir = os.path.join(path.model_path, 'runs')
    save_dir = os.path.join(run_dir, run_id)
    if delete_exist:
        delete_if_exist(save_dir)
    return save_dir


class EstimatorRunner:
    def __init__(self):
        self.modle = None


    def stance_cold(self):
        hp = hyperparams.HPColdStart()
        topic = "atheism"
        setting = shared_setting.TopicTweets2Stance(topic)
        model_dir = get_model_dir("stance_cold_{}".format(topic))

        my_feature_columns = []
        for key in ["features"]:
            my_feature_columns.append(tf.feature_column.numeric_column(key=key))

        task = Classification(3)
        model = Transformer(hp, setting.vocab_size, task)
        param = {
            'feature_columns': my_feature_columns,
            'n_classes': 3,
        }
        estimator = tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=model_dir,
            params=param,
            config=None)

        stance_data = stance_detection.DataLoader(topic, hp.seq_max, setting.vocab_filename)

        def train_input_fn(features, labels, batch_size):
            f_dict = pd.DataFrame(data=features)
            dataset = tf.data.Dataset.from_tensor_slices((f_dict, labels))
            # Shuffle, repeat, and batch the examples.
            return dataset.shuffle(1000).repeat().batch(batch_size)

        def dev_input_fn(batch_size):
            features, labels = stance_data.get_dev_data()
            f_dict = pd.DataFrame(data=features)
            dataset = tf.data.Dataset.from_tensor_slices((f_dict, labels))
            # Shuffle, repeat, and batch the examples.
            return dataset.shuffle(1000).batch(batch_size)

        X, Y = stance_data.get_train_data()
        num_epoch = 10
        batch_size = 32
        step_per_epoch = (len(Y)-1) / batch_size + 1
        tf.logging.info("Logging Test")
        tf.logging.info("num epoch %d", num_epoch)
        estimator.train(lambda:train_input_fn(X, Y, batch_size),
                        max_steps=num_epoch * step_per_epoch)

        print(estimator.evaluate(lambda:dev_input_fn(batch_size)))

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    er = EstimatorRunner()
    er.stance_cold()