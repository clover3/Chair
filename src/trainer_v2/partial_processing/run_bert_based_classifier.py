import functools
import logging
from typing import List

import tensorflow as tf

from trainer_v2.chair_logging import c_log, IgnoreFilter
from trainer_v2.run_config import RunConfigEx
from trainer_v2.train_util.callbacks import get_checkpoint_callback, get_summary_callback
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def load_checkpoint(init_checkpoint: str, checkpoint_type: str,
                    model, sub_module_list: List):
    if not init_checkpoint:
        return

    c_log.info("checkpoint_type: {}".format(checkpoint_type))
    if checkpoint_type == "bert":
        print(sub_module_list)
        for sub_module in sub_module_list:
            c_log.info("Loading model from {}".format(init_checkpoint))
            print("sub_model", sub_module)
            checkpoint = tf.train.Checkpoint(model=sub_module)
            checkpoint.restore(init_checkpoint).assert_existing_objects_matched()
    elif checkpoint_type == "resume":
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(init_checkpoint).assert_existing_objects_matched()
        c_log.debug("checkpoint.restore")
    else:
        raise ValueError("Checkpoint type {} is not expected".format(checkpoint_type))


def ignore_slow_callback():
    ignore_msg =  ["is slow compared to the batch time"]
    ignore_filter = IgnoreFilter(ignore_msg)
    tf_logging = logging.getLogger("tensorflow")
    tf_logging.addFilter(ignore_filter)


def run_keras_fit(get_model_fn, loss_fn, metric_fn,
                  run_config: RunConfigEx,
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
    c_log.debug("run_keras_fit 1")
    if evaluation_dataset:
        validation_steps = 100
    else:
        validation_steps = 0
    c_log.debug("run_keras_fit 2")

    model, sub_model_list = get_model_fn()
    optimizer = model.optimizer
    c_log.debug("run_keras_fit 2.5")
    load_checkpoint(init_checkpoint, run_config.checkpoint_type, model, sub_model_list)
    # load_checkpoint_dev(init_checkpoint, run_config.checkpoint_type, model, sub_model_list)
    c_log.debug("run_keras_fit 3")

    if not isinstance(metric_fn, (list, tuple)):
        metric_fn = [metric_fn]
    c_log.debug("run_keras_fit 4")
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[fn() for fn in metric_fn],
        steps_per_execution=steps_per_loop
    )
    c_log.debug("run_keras_fit 5")
    model.summary()
    c_log.debug("run_keras_fit 6")

    # Prepare to run
    summary_callback = get_summary_callback(model_dir)
    checkpoint_callback = get_checkpoint_callback(model, model_dir, optimizer, run_config)

    if use_callback:
        custom_callbacks = [checkpoint_callback]
        ignore_slow_callback()
    else:
        custom_callbacks = []

    # Run
    c_log.debug("model.fit")
    c_log.debug("training_dataset: {}".format(training_dataset))
    history = model.fit(
        x=training_dataset,
        validation_data=evaluation_dataset,
        validation_steps=validation_steps,
        steps_per_epoch=run_config.steps_per_epoch,
        epochs=epochs,
        callbacks=custom_callbacks
    )
    c_log.info("model.fit completed")
    return history, model


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
        history, outer_model = run_keras_fit(get_model_fn, loss_fn, metric_fn,
                                             run_config, train_input_fn, eval_input_fn)
    stats = {'total_training_steps': run_config.steps_per_epoch * run_config.get_epochs()}

    if 'loss' in history.history:
        stats['train_loss'] = history.history['loss'][-1]
    if 'val_accuracy' in history.history:
        stats['eval_metrics'] = history.history['val_accuracy'][-1]
    return outer_model, stats
