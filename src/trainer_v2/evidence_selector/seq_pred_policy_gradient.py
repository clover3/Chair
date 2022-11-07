from typing import Callable, Tuple, List

import tensorflow as tf

from list_lib import right
from trainer_v2.reinforce.defs import RLStateI
from trainer_v2.reinforce.monte_carlo_policy_function import Action, monte_carlo_explore, PolicyFunction


class RLStateTensor(RLStateI):
    def __init__(self, input_ids, segment_ids):
        self.input_ids = input_ids
        self.segment_ids = segment_ids

    def get_batch_size(self):
        pass


def batch_input_to_state_list(batch_x) -> List[RLStateTensor]:
    input_ids, segment_ids = batch_x
    batch_size = len(input_ids)
    t_list = []
    for i in range(batch_size):
        t = RLStateTensor(input_ids[i], segment_ids[i])
        t_list.append(t)
    return t_list


def batch_rl_state_list(l: List[RLStateTensor]):
    input_ids_l = [e.input_ids for e in l]
    segment_ids_l = [e.segment_ids for e in l]
    return tf.stack(input_ids_l, axis=0), tf.stack(segment_ids_l, axis=0)


# This works as interface (provider) for TF
class SeqPredREINFORCE:
    def __init__(self,
                 seq_length: int,
                 build_state_dataset: Callable[[str, bool], tf.data.Dataset],
                 batch_size,
                 environment,
                 ):
        self.seq_length = seq_length
        self.build_state_dataset = build_state_dataset
        self.batch_size = batch_size
        self.environment = environment
        self.policy_function: PolicyFunction = None

    def init(self, policy_function: PolicyFunction):
        self.policy_function = policy_function

    def get_dataset(self,
                    src_path,
                    is_training,
                    ) -> tf.data.Dataset:
        print("get_dataset")
        RawState = Tuple[tf.Tensor, tf.Tensor]

        def raw_data_iter():
            src_dataset: tf.data.Dataset = self.build_state_dataset(src_path, is_training)
            for batch in src_dataset:
                x, dummy_y = batch
                state_list = batch_input_to_state_list(x)
                items = monte_carlo_explore(self.environment,
                                            self.policy_function,
                                            state_list)
                for sa, sa_list, reward, reward_list in items:
                    rl_state: RLStateTensor = sa[0]
                    state: RawState = rl_state.input_ids, rl_state.segment_ids

                    action: Action = sa[1]
                    sa_list: List[Action] = right(sa_list)
                    yield {
                        'state': state,
                        'action': action,  # np.array [B, L]
                        'sample_actions': sa_list,
                        'base_reward': reward,
                        'sample_reward_list': reward_list,
                    }

        seq_length = self.seq_length
        int_list_spec = tf.TensorSpec(shape=(seq_length,), dtype=tf.int32)
        output_signature = {'state': (int_list_spec, int_list_spec),
                            'action': int_list_spec,
                            'sample_actions': tf.TensorSpec(shape=(None, seq_length,), dtype=tf.int32),
                            'base_reward': tf.TensorSpec(shape=(), dtype=tf.float32),
                            'sample_reward_list': tf.TensorSpec(shape=(None), dtype=tf.float32),
                            }
        dataset = tf.data.Dataset.from_generator(raw_data_iter, output_signature=output_signature)
        dataset = dataset.batch(self.batch_size)
        return dataset
