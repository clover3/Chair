import os
import sys

import numpy as np

from cpath import get_bert_config_path
from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_two_seg_data
from trainer_v2.custom_loop.neural_network_def.asymmetric import ModelConfig2SegProject
from trainer_v2.custom_loop.run_config2 import get_run_config2_nli, RunConfig2
from trainer_v2.train_util.arg_flags import flags_parser
import tensorflow as tf
from keras import backend as K

@report_run3
def main(args):
    c_log.info("Start {}".format(__file__))
    run_config: RunConfig2 = get_run_config2_nli(args)

    model_config = ModelConfig2SegProject()

    def dataset_factory(input_files, is_for_training):
        return get_two_seg_data(input_files, run_config, model_config, is_for_training)
    bert_params = load_bert_config(get_bert_config_path())
    strategy = get_strategy_from_config(run_config)
    eval_step = run_config.eval_config.eval_step
    model_path = run_config.eval_config.model_save_path


    model = tf.keras.models.load_model(model_path)
    eval_dataset = dataset_factory(run_config.dataset_config.eval_files_path, False)
    eval_dataset = eval_dataset.take(10)
    eval_dataset = distribute_dataset(strategy, eval_dataset)
    local_decision_layer = model.layers[10]
    new_outputs = [local_decision_layer.output, model.outputs]
    fun = K.function([model.input, ], new_outputs)  # evaluation function
    tokenizer = get_tokenizer()

    iterator = iter(eval_dataset)
    for batch in iterator:
        x, y = batch
        z, z_label_l = fun(x)
        z_label = z_label_l[0]
        input_ids1, _, input_ids2, _ = x
        for i in range(16):
            pred = np.argmax(z_label[i])
            print("Pred: ", pred, " label :", y[i])
            tokens = tokenizer.convert_ids_to_tokens(input_ids1.numpy()[i])
            print("prem: ", pretty_tokens(tokens, True))
            tokens = tokenizer.convert_ids_to_tokens(input_ids2.numpy()[i][:50])
            print("hypo1: ", pretty_tokens(tokens, True))
            tokens = tokenizer.convert_ids_to_tokens(input_ids2.numpy()[i][50:100])
            print("hypo2: ", pretty_tokens(tokens, True))
            print("local decisions: ", np.argmax(z[i], axis=1))
            print(z[i])
            print()
        input()


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
