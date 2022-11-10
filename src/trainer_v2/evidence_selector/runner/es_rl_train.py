import os
import sys
import tensorflow as tf
from cpath import get_bert_config_path
from models.transformer.bert_common_v2 import get_shape_list2
from typing import List, Callable
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_sequence_labeling_dataset
from trainer_v2.custom_loop.definitions import ModelConfig600_2, ModelConfig300_2
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.custom_loop.modeling_common.network_utils import split_stack_input
from trainer_v2.custom_loop.neural_network_def.seq_pred import SeqPrediction
from trainer_v2.custom_loop.per_task.rl_trainer import PolicyGradientTrainer
from trainer_v2.evidence_selector.enviroment import PEPEnvironment
from trainer_v2.evidence_selector.seq_pred_policy_gradient import SeqPredREINFORCE
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2_nli
from trainer_v2.custom_loop.train_loop import tf_run
from trainer_v2.evidence_selector.policy_function_for_evidence_selector import SequenceLabelPolicyFunction
from trainer_v2.train_util.arg_flags import flags_parser


# Train Evidence Selector with Reinforcement Learning. REINFORCE method

def main(args):
    c_log.info("Start Train es_rl_train.py")
    bert_params = load_bert_config(get_bert_config_path())
    src_model_config = ModelConfig600_2()
    num_window = 2
    window_length = int(src_model_config.max_seq_length / num_window)
    def build_state_dataset(input_files, is_for_training):
        dataset = get_sequence_labeling_dataset(input_files, run_config, src_model_config, is_for_training)

        def split_stack(x, y):
            input_ids, segment_ids = x
            label_ids = y
            input_list = [input_ids, segment_ids, label_ids]
            input_list_stacked = split_stack_input(input_list,
                                                   src_model_config.max_seq_length,
                                                   window_length)
            batch_size, _ = get_shape_list2(x[0])

            def r3to2(arr):
                return tf.reshape(arr, [batch_size * num_window, window_length])

            input_list_flatten = list(map(r3to2, input_list_stacked))  # [batch_size * num_window, window_length]
            input_ids, segment_ids, label_ids = input_list_flatten
            return (input_ids, segment_ids), label_ids

        dataset = dataset.map(split_stack,
                              num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    model_config = ModelConfig300_2()
    task_model = SeqPrediction()
    server = "localhost"
    if "PEP_SERVER" in os.environ:
        server = os.environ["PEP_SERVER"]
    c_log.info("PEP_SERVER: {}".format(server))
    pep_env = PEPEnvironment(server)
    run_config: RunConfig2 = get_run_config2_nli(args)
    run_config.print_info()
    window_length = model_config.max_seq_length
    reinforce = SeqPredREINFORCE(window_length, build_state_dataset,
                                 run_config.common_run_config.batch_size,
                                 pep_env.get_item_results,
                                 )

    trainer: PolicyGradientTrainer = PolicyGradientTrainer(bert_params,
                                                           model_config,
                                                           run_config,
                                                           task_model,
                                                           SequenceLabelPolicyFunction,
                                                           reinforce
                                                           )
    tf_run(run_config, trainer, trainer.build_dataset)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


