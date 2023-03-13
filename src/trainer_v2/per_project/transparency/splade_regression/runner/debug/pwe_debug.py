

import sys

from transformers import AutoTokenizer
import tensorflow as tf

from taskman_client.task_proxy import get_task_manager_proxy, get_local_machine_name
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.per_project.transparency.splade_regression.data_loaders.dataset_factories import get_three_text_dataset
from trainer_v2.per_project.transparency.splade_regression.data_loaders.pairwise_eval import load_pairwise_eval_data, \
    PairwiseEval, build_pairwise_eval_dataset
from trainer_v2.per_project.transparency.splade_regression.modeling.regression_modeling import get_regression_model
from trainer_v2.train_util.arg_flags import flags_parser
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def dict_to_tuple(features):
    t_list = []
    for idx in range(3):
        t = features[f"input_ids_{idx}"], features[f"attention_mask_{idx}"]
        t_list.append(t)
    return tuple(t_list)


def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2(args)

    model_config = {
        "model_type": "distilbert-base-uncased",
        "max_seq_length": 256
    }
    c_log.info("loading model from %s", run_config.eval_config.model_save_path)
    model = tf.keras.models.load_model(run_config.eval_config.model_save_path)
    dataset: tf.data.Dataset = get_three_text_dataset(
        run_config.dataset_config.eval_files_path, model_config, run_config, False
    )
    dataset = dataset.map(dict_to_tuple)

    itr = iter(dataset)

    for _ in range(2):
        q, d1, d2 = next(itr)
        q_enc = model(q, training=False)
        d1_enc = model(d1, training=False)
        d2_enc = model(d2, training=False)
        print(d1_enc.shape)
        def score(q_enc, d_enc):
            return tf.reduce_sum(tf.multiply(q_enc, d_enc), axis=1)

        s1 = score(q_enc, d1_enc)
        s2 = score(q_enc, d2_enc)

        print(s1, s2)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


