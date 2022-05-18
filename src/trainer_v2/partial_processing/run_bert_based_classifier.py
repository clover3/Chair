import functools
import logging
from typing import List

import tensorflow as tf

from trainer_v2.chair_logging import c_log, IgnoreFilter
from trainer_v2.kera_debug.dev_name_mapping import normalize_mem_var_inner
from trainer_v2.kera_debug.name_based_checkpoint_loader import load_stock_weights
from trainer_v2.partial_processing.bert_encoder_module import BertEncoderLayer
from trainer_v2.run_config import RunConfigEx
from trainer_v2.train_util.callbacks import get_checkpoint_callback, get_summary_callback
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def get_common_prefix(names):
    """
    :type strs: List[str]
    :rtype: str
    """
    if len(names) == 0:
        return ""
    current = names[0]
    for i in range(1, len(names)):
        temp = ""
        if len(current) == 0:
            break
        for j in range(len(names[i])):
            if j < len(current) and current[j] == names[i][j]:
                temp += current[j]
            else:
                break
        current = temp
    return current


def load_checkpoint(init_checkpoint: str, checkpoint_type: str,
                    model, sub_module_list: List):
    if not init_checkpoint:
        return

    c_log.info("checkpoint_type: {}".format(checkpoint_type))
    if checkpoint_type == "bert":
        for i, sub_module in enumerate(sub_module_list):
            load_from_reshaped_bert_checkpoint(sub_module, init_checkpoint)
    elif checkpoint_type == "resume":
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(init_checkpoint).assert_existing_objects_matched()
        c_log.debug("checkpoint.restore")
    else:
        raise ValueError("Checkpoint type {} is not expected".format(checkpoint_type))


def load_from_reshaped_bert_checkpoint(sub_module, init_checkpoint):
    # prefix = f"bert_encoder_layer/encoder{i+1}"
    prefix = get_common_prefix([v.name for v in sub_module.variables])
    if not sub_module.variables:
        c_log.error("sub module has no variables")

    def name_mapping(name):
        return "/".join(normalize_mem_var_inner(prefix, name))

    c_log.info("Loading model from {}".format(init_checkpoint))
    show_sub_model_param(sub_module)
    load_stock_weights(sub_module, init_checkpoint, name_mapping, "cls", 199)
    show_sub_model_param(sub_module)


def ignore_slow_callback():
    ignore_msg = ["is slow compared to the batch time"]
    ignore_filter = IgnoreFilter(ignore_msg)
    tf_logging = logging.getLogger("tensorflow")
    tf_logging.addFilter(ignore_filter)


def show_sub_model_param(sub_model: BertEncoderLayer):
    embedding = sub_model.get_embedding_table()
    target_idx = 802
    vector = embedding[target_idx].numpy().tolist()
    print(vector[:10])
    # save_to_pickle(vector, "debug_embedding_802")


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
    if evaluation_dataset:
        validation_steps = 100
    else:
        validation_steps = 0

    model, sub_model_list = get_model_fn()
    model: tf.keras.Model = model
    optimizer = model.optimizer
    print("Before")
    load_checkpoint(init_checkpoint, run_config.checkpoint_type, model, sub_model_list)
    print("After")

    if not isinstance(metric_fn, (list, tuple)):
        metric_fn = [metric_fn]
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[fn() for fn in metric_fn],
        steps_per_execution=steps_per_loop
    )
    model.summary(line_length=100)

    # Prepare to run
    summary_callback = get_summary_callback(model_dir)
    checkpoint_callback = get_checkpoint_callback(model, model_dir, optimizer, run_config)

    if use_callback:
        custom_callbacks = [checkpoint_callback]
        ignore_slow_callback()
    else:
        custom_callbacks = []

    # Run
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
