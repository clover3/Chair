import sys

import tensorflow as tf

from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import create_dataset_common
from trainer_v2.custom_loop.modeling_common.adam_decay import AdamWeightDecay
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset, apply_gradient_warning_less
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.per_project.transparency.splade_regression.data_loaders.dataset_factories import \
    get_dummy_vector_regression_dataset
from trainer_v2.per_project.transparency.splade_regression.modeling.regression_modeling import \
    get_dummy_regression_model
from trainer_v2.train_util.arg_flags import flags_parser



from transformers import TFAutoModelForSequenceClassification
def main(args):
    run_config: RunConfig2 = get_run_config2(args)
    #
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    model_checkpoint = "distilbert-base-cased"
    seq_length = 300
    def decode_record(record):
        name_to_features = {
            'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
            'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'label_ids': tf.io.FixedLenFeature([], tf.int64),
        }
        d = tf.io.parse_single_example(record, name_to_features)

        X = {
            "input_ids": tf.zeros([seq_length, ], tf.int32),
            "attention_mask": tf.zeros([seq_length, ], tf.int32),
        }
        Y = tf.zeros([30522], tf.float32)

        return X, Y

    dataset = create_dataset_common(decode_record, run_config,
                                    run_config.dataset_config.train_files_path, False)
    dist_train_dataset = distribute_dataset(strategy, dataset)
    optimizer = AdamWeightDecay(
        learning_rate=run_config.train_config.learning_rate,
        exclude_from_weight_decay=[]
    )
    c_log.info("Use tf.keras.losses.MeanSquaredError as loss ")
    loss_fn_inner = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    # loss = "MSE"
    def loss_fn(labels, predictions):
        per_example_loss = loss_fn_inner(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=16)

    with strategy.scope():
        model = get_dummy_regression_model(300, None)
        # model.compile(loss=loss, optimizer=optimizer)
        # model.fit(dataset, steps_per_epoch=1000)

        def train_step(item):
            x, y = item
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)
                loss = loss_fn(y, prediction)

            gradients = tape.gradient(loss, model.trainable_variables)
            apply_gradient_warning_less(optimizer, gradients, model.trainable_variables)
            return loss
        #

        model.compile(optimizer="adam")
        train_itr = iter(dist_train_dataset)
        @tf.function
        def distributed_train_step(train_itr, steps_per_execution):
            # try:
            total_loss = 0.0
            n_step = 0.
            for _ in tf.range(steps_per_execution):
                batch_item = next(train_itr)
                per_replica_losses = strategy.run(train_step, args=(batch_item, ))
                loss = strategy.reduce(
                    tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
                total_loss += loss
                n_step += 1.

            train_loss = total_loss / n_step
            return train_loss

        train_loss = distributed_train_step(train_itr, 1)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


