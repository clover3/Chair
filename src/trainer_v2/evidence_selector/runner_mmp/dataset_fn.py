
# Train Evidence Selector with Reinforcement Learning. REINFORCE method
from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.custom_loop.dataset_factories import get_classification_dataset_hf_to_bert_f2
from trainer_v2.custom_loop.modeling_common.network_utils import split_stack_input
import tensorflow as tf



class SplitStack:
    def __init__(self, src_max_seq_length):
        num_window = 2
        window_length = int(src_max_seq_length / num_window)

        self.window_length = window_length
        self.num_window = num_window
        self.src_max_seq_length = src_max_seq_length

    def apply_to_xy(self, x, y):
        input_ids, segment_ids = x
        new_input_ids, new_segment_ids = self.apply(input_ids, segment_ids)
        return (new_input_ids, new_segment_ids), y

    def apply(self, input_ids, segment_ids):
        window_length = self.window_length
        input_list = [input_ids, segment_ids]
        input_list_stacked = split_stack_input(
            input_list,
            self.src_max_seq_length,
            self.window_length)
        batch_size, _ = get_shape_list2(input_ids)

        def r3to2(arr):
            return tf.reshape(arr, [batch_size * self.num_window, window_length])

        input_list_flatten = list(map(r3to2, input_list_stacked))  # [batch_size * num_window, window_length]
        input_ids, segment_ids = input_list_flatten
        return input_ids, segment_ids


def build_state_dataset_fn(run_config, src_model_config):
    split_stack_module = SplitStack(src_model_config.max_seq_length)

    def build_state_dataset(input_files, is_for_training):
        dataset = get_classification_dataset_hf_to_bert_f2(input_files, run_config, src_model_config, is_for_training)
        dataset = dataset.map(split_stack_module.apply_to_xy,
                              num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    return build_state_dataset
