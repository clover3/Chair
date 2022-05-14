import functools
import os

import tensorflow as tf
from official.nlp.bert import configs as bert_configs
from official.utils.misc import keras_utils

from cpath import get_bert_config_path
from trainer_v2.chair_logging import c_log
from trainer_v2.get_tpu_strategy import get_strategy
from trainer_v2.partial_processing.config_helper import ModelConfig
from trainer_v2.partial_processing.misc_helper import get_custom_callback


def run_keras_compile_fit(model_dir,
                          strategy,
                          model_fn,
                          train_input_fn,
                          eval_input_fn,
                          loss_fn,
                          metric_fn,
                          init_checkpoint,
                          epochs,
                          steps_per_epoch,
                          steps_per_loop,
                          eval_steps,
                          training_callbacks=True,
                          custom_callbacks=None):
    """Runs BERT classifier model using Keras compile/fit API."""

    with strategy.scope():
        training_dataset = train_input_fn()
        evaluation_dataset = eval_input_fn() if eval_input_fn else None
        outer_model, sub_model_list = model_fn()
        optimizer = outer_model.optimizer

        if init_checkpoint:
            for sub_model in sub_model_list:
                checkpoint = tf.train.Checkpoint(model=sub_model)
                checkpoint.restore(init_checkpoint).assert_existing_objects_matched()

        if not isinstance(metric_fn, (list, tuple)):
            metric_fn = [metric_fn]
        outer_model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[fn() for fn in metric_fn],
            steps_per_execution=steps_per_loop)

        outer_model.summary()

        summary_dir = os.path.join(model_dir, 'summaries')
        summary_callback = tf.keras.callbacks.TensorBoard(summary_dir)
        checkpoint = tf.train.Checkpoint(model=outer_model, optimizer=optimizer)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=model_dir,
            max_to_keep=None,
            step_counter=optimizer.iterations,
            checkpoint_interval=0)
        checkpoint_callback = keras_utils.SimpleCheckpoint(checkpoint_manager)

        if training_callbacks:
            if custom_callbacks is not None:
                custom_callbacks += [summary_callback, checkpoint_callback]
            else:
                custom_callbacks = [summary_callback, checkpoint_callback]

        history = outer_model.fit(
            x=training_dataset,
            validation_data=evaluation_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_steps=eval_steps,
            callbacks=custom_callbacks)
        stats = {'total_training_steps': steps_per_epoch * epochs}
        if 'loss' in history.history:
            stats['train_loss'] = history.history['loss'][-1]
        if 'val_accuracy' in history.history:
            stats['eval_metrics'] = history.history['val_accuracy'][-1]
        return outer_model, stats


def get_model_config():
    bert_config = get_bert_config()
    max_seq_length = 300
    model_config = ModelConfig(bert_config, max_seq_length)
    return model_config


def get_bert_config():
    bert_config = bert_configs.BertConfig.from_json_file(get_bert_config_path())
    return bert_config


def run_keras_compile_fit_wrap(args, get_model_fn, run_config, train_input_fn):
    strategy = get_strategy(args.use_tpu, args.tpu_name)
    c_log.info("run_fn entry")
    metric_fn = functools.partial(
        tf.keras.metrics.SparseCategoricalAccuracy,
        'accuracy',
        dtype=tf.float32)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    run_keras_compile_fit(
        args.output_dir,
        strategy,
        get_model_fn,
        train_input_fn,
        None,
        loss_fn,
        metric_fn,
        run_config.init_checkpoint,
        run_config.get_epochs(),
        run_config.steps_per_epoch,
        steps_per_loop=run_config.steps_per_execution,
        eval_steps=0,
        training_callbacks=get_custom_callback(run_config),
    )
