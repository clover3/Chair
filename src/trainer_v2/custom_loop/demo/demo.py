import os
import sys

import numpy as np

from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig200_200
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_two_seg_data
from trainer_v2.custom_loop.run_config2 import get_run_config2_nli, RunConfig2
from trainer_v2.train_util.arg_flags import flags_parser
import tensorflow as tf
from keras import backend as K


def load_local_decision_nli(model_path):
    model = tf.keras.models.load_model(model_path)
    local_decision_layer_idx = 12
    local_decision_layer = model.layers[local_decision_layer_idx]
    print("Local decision layer", local_decision_layer.name)
    new_outputs = [local_decision_layer.output, model.outputs]
    fun = K.function([model.input, ], new_outputs)  # evaluation function
    return fun


@report_run3
def main(args):
    c_log.info("Start {}".format(__file__))
    run_config: RunConfig2 = get_run_config2_nli(args)

    model_config = ModelConfig200_200()

    strategy = get_strategy_from_config(run_config)
    model_path = run_config.eval_config.model_save_path

    fun = load_local_decision_nli(model_path)

    def dataset_factory(input_files, is_for_training):
        return get_two_seg_data(input_files, run_config, model_config, is_for_training)

    tokenizer = get_tokenizer()
    eval_dataset = dataset_factory(run_config.dataset_config.eval_files_path, False)
    eval_dataset = eval_dataset.take(10)
    eval_dataset = distribute_dataset(strategy, eval_dataset)

    batch_size = run_config.common_run_config.batch_size
    iterator = iter(eval_dataset)
    for batch in iterator:
        x, y = batch
        z, z_label_l = fun(x)
        z_label = z_label_l[0]
        input_ids1, _, input_ids2, _ = x
        for i in range(batch_size):
            pred = np.argmax(z_label[i])
            print("Pred: ", pred, " label :", y[i])
            tokens = tokenizer.convert_ids_to_tokens(input_ids1.numpy()[i])
            print("prem: ", pretty_tokens(tokens, True))
            input_ids2_np = input_ids2.numpy()[i]
            tokens = tokenizer.convert_ids_to_tokens(input_ids2_np[:100])
            print("hypo1: ", pretty_tokens(tokens, True))
            tokens = tokenizer.convert_ids_to_tokens(input_ids2_np[100:])
            print("hypo2: ", pretty_tokens(tokens, True))
            print("local decisions: ", np.argmax(z[i], axis=1))
            print(z[i])
            print()
        input("Press enter to continue")


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
