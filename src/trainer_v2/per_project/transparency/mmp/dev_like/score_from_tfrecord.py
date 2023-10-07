import sys

import yaml

from table_lib import tsv_iter
from misc_lib import select_third_fourth
from taskman_client.task_proxy import get_task_manager_proxy
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_pointwise
from trainer_v2.custom_loop.definitions import ModelConfig256_1
from trainer_v2.custom_loop.run_config2 import get_run_config_for_predict_empty
from trainer_v2.per_project.transparency.mmp.eval_helper.mmp_eval_line_format import eval_dev_mrr, \
    predict_and_batch_save_scores
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import batch_score_and_save_score_lines
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank import get_scorer, build_inference_model2
from trainer_v2.train_util.arg_flags import flags_parser
from trainer_v2.train_util.get_tpu_strategy import get_strategy
from typing import List, Iterable, Callable, Dict, Tuple, Set
import tensorflow as tf


def main():
    with open(sys.argv[1], 'r') as file:
        config = yaml.safe_load(file)

    tfrecord_path = config['tfrecord_path']
    model_path = config['model_path']
    scores_path = config['score_save_path']
    use_tpu = "tpu_name" in config
    if not "tpu_name" in config:
        config["tpu_name"] = ""

    run_config = get_run_config_for_predict_empty()

    def build_dataset(input_files, is_for_training):
        return get_pointwise(
            input_files, run_config, ModelConfig256_1(), is_for_training)


    strategy = get_strategy(use_tpu, config['tpu_name'])
    c_log.info("Start prediction")
    with strategy.scope():
        paired_model = tf.keras.models.load_model(model_path, compile=False)
        inference_model = build_inference_model2(paired_model)
        dataset = build_dataset(tfrecord_path, False)
        output = inference_model.predict(dataset)

    c_log.info("Done")

    return NotImplemented


if __name__ == "__main__":
    main()