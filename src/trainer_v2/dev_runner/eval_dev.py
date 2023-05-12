from typing import Dict

import tensorflow as tf
from tensorflow.keras import layers


# Build a simple neural network model
from cpath import get_canonical_model_path
from trainer_v2.custom_loop.eval_loop import tf_run_eval
from trainer_v2.custom_loop.evaler_if import EvalerIF
from trainer_v2.custom_loop.run_config2 import RunConfig2, CommonRunConfig, EvalConfig, DatasetConfig


def build_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# Define the dataset
def build_dataset():
    # Generate some random data
    x_train = tf.random.normal(shape=(1000, 100))
    y_train = tf.random.uniform(shape=(1000,), maxval=2, dtype=tf.int32)
    x_test = tf.random.normal(shape=(200, 100))
    y_test = tf.random.uniform(shape=(200,), maxval=2, dtype=tf.int32)

    # Convert labels to one-hot encoding

    # Create the training and testing datasets
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=1000).batch(batch_size=32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(batch_size=32)
    print(train_ds)
    return train_ds, test_ds


# Train the model
def train_model(model, train_ds, test_ds, epochs):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    metrics = [tf.keras.metrics.CategoricalAccuracy()]
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=metrics)

    model.fit(train_ds, epochs=epochs)

    test_loss, test_acc = model.evaluate(test_ds)
    model_save_path = get_canonical_model_path("eval_dev")
    model.save(model_save_path)
    print('Test accuracy:', test_acc)


def train_with_fit():
    # Set the hyperparameters
    input_shape = (100,)
    num_classes = 2
    epochs = 10

    # Build the model and dataset
    model = build_model(input_shape, num_classes)
    train_ds, test_ds = build_dataset()

    # Train the model on the dataset
    train_model(model, train_ds, test_ds, epochs)


class EvalerDOut(EvalerIF):
    def build(self, model):
        self.model = model
        pass

    def get_keras_model(self) -> tf.keras.Model:
        return self.model

    def eval_fn(self, item):
        model = self.get_keras_model()
        output_d = model(item, training=False)

    def get_eval_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        return self.eval_metrics


def eval_with_tf_run_eval():
    run_config = RunConfig2(
        CommonRunConfig(),
        DatasetConfig("", ""),
        eval_config=EvalConfig()
    )

    def build_dataset_dummy(path, is_training):
        x_test = tf.random.normal(shape=(200, 100))
        y_test = tf.random.uniform(shape=(200,), maxval=2, dtype=tf.int32)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_ds = test_ds.batch(batch_size=32)
        return test_ds
    evaler = EvalerDOut()
    tf_run_eval(run_config, evaler, build_dataset_dummy)


def main():
    eval_with_tf_run_eval()


if __name__ == "__main__":
    main()