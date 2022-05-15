import functools
import os

import tensorflow as tf
from official.utils.misc import keras_utils

from trainer_v2.chair_logging import c_log
from trainer_v2.get_tpu_strategy import get_strategy


def load_checkpoint(init_checkpoint, sub_model_list):
    if init_checkpoint:
        for sub_model in sub_model_list:
            c_log.info("Loading model from {}".format(init_checkpoint))
            checkpoint = tf.train.Checkpoint(model=sub_model)
            checkpoint.restore(init_checkpoint).assert_existing_objects_matched()


def run_keras_fit(get_model_fn, loss_fn, metric_fn, run_config,
                  train_input_fn, eval_input_fn):
    c_log.debug("run_keras_fit ENTRY")
    # List parameters
    init_checkpoint = run_config.init_checkpoint
    steps_per_loop = run_config.steps_per_execution
    model_dir = run_config.model_save_path
    use_callback = True
    epochs = run_config.get_epochs()

    # Initialize dataset and model
    training_dataset = train_input_fn()
    evaluation_dataset = eval_input_fn() if eval_input_fn is not None else None
    model, sub_model_list = get_model_fn()
    optimizer = model.optimizer

    load_checkpoint(init_checkpoint, sub_model_list)

    if not isinstance(metric_fn, (list, tuple)):
        metric_fn = [metric_fn]
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[fn() for fn in metric_fn],
        steps_per_execution=steps_per_loop
    )
    model.summary()

    # Prepare to run
    summary_callback = get_summary_callback(model_dir)
    checkpoint_callback = get_checkpoint_callback(model, model_dir, optimizer)

    if use_callback:
        custom_callbacks = [summary_callback, checkpoint_callback]
    else:
        custom_callbacks = []

    # Run
    history = model.fit(
        x=training_dataset,
        validation_data=evaluation_dataset,
        steps_per_epoch=run_config.steps_per_epoch,
        epochs=epochs,
        callbacks=custom_callbacks
    )
    c_log.info("model.fit completed")
    return history, model


def get_checkpoint_callback(model, model_dir, optimizer):
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=model_dir,
        max_to_keep=None,
        step_counter=optimizer.iterations,
        checkpoint_interval=10000)
    checkpoint_callback = keras_utils.SimpleCheckpoint(checkpoint_manager)
    return checkpoint_callback


def get_summary_callback(model_dir):
    summary_dir = os.path.join(model_dir, 'summaries')
    summary_callback = tf.keras.callbacks.TensorBoard(summary_dir, update_freq=1)
    return summary_callback


def run_classification(args,
                       run_config,
                       get_model_fn,
                       train_input_fn,
                       eval_input_fn):
    c_log.info("run_classification entry")
    strategy = get_strategy(args.use_tpu, args.tpu_name)
    metric_fn = functools.partial(tf.keras.metrics.SparseCategoricalAccuracy,
                                  'accuracy',
                                  dtype=tf.float32)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    with strategy.scope():
        history, outer_model = run_keras_fit(get_model_fn, loss_fn, metric_fn, run_config, train_input_fn, eval_input_fn)
    stats = {'total_training_steps': run_config.steps_per_epoch * run_config.get_epochs()}

    if 'loss' in history.history:
        stats['train_loss'] = history.history['loss'][-1]
    if 'val_accuracy' in history.history:
        stats['eval_metrics'] = history.history['val_accuracy'][-1]
    return outer_model, stats
