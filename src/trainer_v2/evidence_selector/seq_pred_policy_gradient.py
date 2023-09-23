from typing import Callable, Tuple, List, NamedTuple

import tensorflow as tf

from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens
from list_lib import right, lmap
from misc_lib import NamedAverager, two_digit_float, tensor_to_list
from trainer.promise import PromiseKeeper, MyPromise, MyFuture, list_future
from trainer_v2.chair_logging import c_log
from trainer_v2.evidence_selector.defs import RLStateTensor
from trainer_v2.evidence_selector.enviroment import PEInfoI
from trainer_v2.reinforce.monte_carlo_policy_function import Action, PolicyFunction
import numpy as np


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


SA = Tuple[RLStateTensor, Action]


class ExplorationReward(NamedTuple):
    base_sa: Tuple[RLStateTensor, Action]
    alt_sa_list: List[Tuple[RLStateTensor, Action]]
    base_reward: float
    alt_reward_list: List[float]


class ExplorationFuture(NamedTuple):
    base_sa: Tuple[RLStateTensor, Action]
    alt_sa_list: List[Tuple[RLStateTensor, Action]]
    base_sa_future: MyFuture
    alt_sa_future_list: List[MyFuture]


class ExplorationOutput(NamedTuple):
    base_sa: Tuple[RLStateTensor, Action]
    alt_sa_list: List[Tuple[RLStateTensor, Action]]
    base_result: PEInfoI
    alt_result_list: List[PEInfoI]


    @classmethod
    def from_future(cls, e: ExplorationFuture):
        return ExplorationOutput(e.base_sa, e.alt_sa_list, e.base_sa_future.get(),
                                 list_future(e.alt_sa_future_list))


class SeqPredREINFORCE:
    def __init__(self,
                 seq_length: int,
                 build_state_dataset: Callable[[str, bool], tf.data.Dataset],
                 batch_size,
                 environment: Callable[[List[Tuple[RLStateTensor, List[int]]]], List[PEInfoI]]
                 ):
        self.seq_length = seq_length
        self.build_state_dataset = build_state_dataset
        self.batch_size = batch_size
        self.environment = environment
        self.policy_function: PolicyFunction = None
        self.summary_callback = None
        self.tokenizer = get_tokenizer()

    def init(self, policy_function: PolicyFunction, summary_callback=None):
        self.policy_function = policy_function
        self.summary_callback = summary_callback

    def get_dataset(self,
                    src_path,
                    is_training,
                    ) -> tf.data.Dataset:
        RawState = Tuple[tf.Tensor, tf.Tensor]

        def raw_data_iter():
            src_dataset: tf.data.Dataset = self.build_state_dataset(src_path, is_training)
            for batch in src_dataset:
                x, dummy_y = batch
                state_list: List[RLStateTensor] = batch_input_to_state_list(x)
                c_log.debug("Before Monte Carlo")
                items: List[ExplorationReward] = self.monte_carlo_explore(state_list)
                c_log.debug("After Monte Carlo")
                for e in items:
                    rl_state: RLStateTensor = e.base_sa[0]
                    state: RawState = rl_state.input_ids, rl_state.segment_ids

                    action: Action = e.base_sa[1]
                    sa_list: List[Action] = right(e.alt_sa_list)
                    yield {
                        'state': state,
                        'action': action,  # np.array [B, L]
                        'sample_actions': sa_list,
                        'base_reward': e.base_reward,
                        'sample_reward_list': e.alt_reward_list,
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
        dataset = dataset.batch(self.batch_size * 2)
        return dataset

    def monte_carlo_explore(self, state_list: List[RLStateTensor])\
            -> List[ExplorationReward]:
        c_log.debug("monte_carlo_explore")
        policy_function = self.policy_function
        pk = PromiseKeeper(self.environment)

        def get_env_result_future(sa: Tuple[RLStateTensor, Action]) -> MyFuture:
            return MyPromise(sa, pk).future()

        batch_size = len(state_list)
        c_log.debug(" - Computing mean actions")
        y_hat: List[Action] = policy_function.get_mean_action(state_list)
        c_log.debug(" - Sampling actions")
        y_s_list: List[List[Action]] = policy_function.sample_actions(state_list)

        y_hat: List[Action] = policy_function.get_top_k_action(state_list)
        c_log.debug(" - building action candidates")

        e_list: List[ExplorationFuture] = []
        assert len(y_hat) == len(y_s_list)

        for i in range(batch_size):
            x = state_list[i]
            sa: Tuple[RLStateTensor, Action] = x, y_hat[i]
            sa_list: List[Tuple[RLStateTensor, Action]] = [(x, y_s) for y_s in y_s_list[i]]
            env_run_future = get_env_result_future(sa)
            erf_list = [get_env_result_future(sa) for sa in sa_list]
            e: ExplorationFuture = ExplorationFuture(sa, sa_list, env_run_future, erf_list)
            e_list.append(e)

        def exploration_output_to_reward(e: ExplorationOutput):
            scores = [a.combined_score() for a in e.alt_result_list]
            return ExplorationReward(e.base_sa, e.alt_sa_list, e.base_result.combined_score(), scores)

        c_log.debug(" - Running action candidates")
        pk.do_duty()
        c_log.debug("monte_carlo_explore DONE")

        e_out_list: List[ExplorationOutput] = lmap(ExplorationOutput.from_future, e_list)
        self.print_stats(e_out_list)
        self.print_items(e_out_list)

        return lmap(exploration_output_to_reward, e_out_list)

    def print_items(self, e_out_list):
        def get_masked_input(input_ids, segment_ids, action):
            select_mask_for_p_tokens: List[int] = action
            select_mask_np = np.array(select_mask_for_p_tokens, np.bool)
            select_mask_np = np.logical_or(select_mask_np, np.array(segment_ids, np.bool))
            select_mask_np = np.logical_or(select_mask_np, np.equal(input_ids, 0))
            input_ids_np = np.array(input_ids, np.int)
            new_input_ids = input_ids_np * select_mask_np + (1 - select_mask_np) * 103
            return new_input_ids

        for exploration_output in e_out_list:
            eo: ExplorationOutput = exploration_output
            state, _ = eo.base_sa

            def to_text(ids):
                t = tensor_to_list(ids)
                return pretty_tokens(self.tokenizer.convert_ids_to_tokens(t), True)

            s = to_text(state.input_ids)
            print(s)
            for item, res in zip(eo.alt_sa_list, eo.alt_result_list):
                state, action = item
                stat_str = " ".join(map(two_digit_float, [res.get_error(), res.density(), res.combined_score()]))
                input_ids = get_masked_input(state.input_ids, state.segment_ids, action)
                s = to_text(input_ids)
                print("{}\t{}".format(stat_str, s))


    def print_stats(self, e_out_list):
        na = NamedAverager()
        for exploration_output in e_out_list:
            base = exploration_output.base_result
            base_error = base.get_error()
            na.avg_dict['base_error'].append(base_error)
            na.avg_dict['base_density'].append((base.density()))
            na.avg_dict['base_reward'].append((base.combined_score()))

            for alt in exploration_output.alt_result_list:
                alt_error = alt.get_error()
                na.avg_dict['alt_error'].append(alt_error)
                na.avg_dict['alt_density'].append(alt.density())
                na.avg_dict['alt_reward'].append((alt.combined_score()))

                na.avg_dict['alt_ce_better'].append(int(alt_error < base_error))
                na.avg_dict['alt_density_better'].append(int(alt.density() < base.density()))
                na.avg_dict['alt_reward_better'].append(int(alt.combined_score() > base.combined_score()))

        ad = na.get_average_dict()
        for part in ['base', 'alt']:
            msg = "{0}[Error/Density/Reward] = {1:.2f} / {2:.2f} / {3:.2f}".format(part,
                                                                              ad[part + "_error"],
                                                                              ad[part + "_density"],
                                                                              ad[part + "_reward"])
            c_log.info(msg)

        msg = "alt better [Error/Density/Reward] = {0:.2f} / {1:.2f} / {2:.2f}".format(
            ad['alt_ce_better'], ad['alt_density_better'], ad['alt_reward_better']
        )
        c_log.info(msg)
        self.summary_callback(ad)
