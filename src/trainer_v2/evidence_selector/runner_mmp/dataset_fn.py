
# Train Evidence Selector with Reinforcement Learning. REINFORCE method
from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.custom_loop.dataset_factories import get_classification_dataset_hf_to_bert_f2
from trainer_v2.custom_loop.modeling_common.network_utils import split_stack_input
import tensorflow as tf

def build_state_dataset_fn(run_config, src_model_config):
    num_window = 2
    window_length = int(src_model_config.max_seq_length / num_window)

    def build_state_dataset(input_files, is_for_training):
        dataset = get_classification_dataset_hf_to_bert_f2(input_files, run_config, src_model_config, is_for_training)

        def split_stack(x, y):
            input_ids, segment_ids = x
            label_ids = y
            input_list = [input_ids, segment_ids]
            input_list_stacked = split_stack_input(input_list,
                                                   src_model_config.max_seq_length,
                                                   window_length)
            batch_size, _ = get_shape_list2(x[0])

            def r3to2(arr):
                return tf.reshape(arr, [batch_size * num_window, window_length])

            input_list_flatten = list(map(r3to2, input_list_stacked))  # [batch_size * num_window, window_length]
            input_ids, segment_ids = input_list_flatten
            return (input_ids, segment_ids), label_ids

        dataset = dataset.map(split_stack,
                              num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    return build_state_dataset

