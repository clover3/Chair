import sys
from typing import Callable
from typing import List, Iterable, Callable, Dict, Tuple, Set

import tensorflow as tf

from trainer_v2.custom_loop.dataset_factories import create_dataset_common, parse_file_path
from trainer_v2.custom_loop.modeling_common.adam_decay import AdamWeightDecay
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.train_util.arg_flags import flags_parser


def get_model():
    x = tf.keras.layers.Input(shape=(10,), dtype='float32', name="x")
    output = tf.keras.layers.Dense(1)(x)
    new_model = tf.keras.models.Model(inputs=[x], outputs=[output])
    return new_model



def create_dataset_common(decode_record: Callable,
                          file_path: str,
                          ):
    input_files: List[str] = parse_file_path(file_path)
    dataset = tf.data.TFRecordDataset(input_files, num_parallel_reads=len(input_files))
    dataset = dataset.map(decode_record,
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(16, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def main():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

    def decode_record(record):
        name_to_features = {
            "x": tf.io.FixedLenFeature([10], tf.float32),
            "y": tf.io.FixedLenFeature([1], tf.float32),
        }
        d = tf.io.parse_single_example(record, name_to_features)
        return d['x'], d['y']

    dataset = create_dataset_common(decode_record, "debug_random_float")
    loss_fn_inner = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    optimizer = AdamWeightDecay(
        learning_rate=1e-5,
        exclude_from_weight_decay=[]
    )

    with strategy.scope():
        model = get_model()
        model.compile(loss='MSE', optimizer='adam')
        model.fit(dataset, steps_per_epoch=10)


if __name__ == "__main__":
    main()


