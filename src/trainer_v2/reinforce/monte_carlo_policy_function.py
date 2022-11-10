from abc import ABC, abstractmethod
from typing import TypeVar
from typing import List, Callable, Tuple
import tensorflow as tf

from trainer.promise import PromiseKeeper, MyPromise, MyFuture, list_future
from trainer_v2.chair_logging import c_log
from trainer_v2.reinforce.defs import RLStateI

Action = TypeVar("Action")
State = TypeVar("State")


# MCPG: Monte-Carlo Policy Gradient

# PolicyFunction should be aware of batch structure
class PolicyFunction(ABC):
    @abstractmethod
    def sample_actions(self, state_list: List[RLStateI]) -> List[List[Action]]:
        pass

    @abstractmethod
    def get_mean_action(self, state_list: List[RLStateI]) -> List[Action]:
        pass

    @abstractmethod
    def get_log_action_prob(self, state_list: List[RLStateI], action_ll: List[List[Action]]) -> tf.Tensor:
        # [Batch, Number of Samples]
        pass


SA = Tuple[State, Action]


def monte_carlo_explore(
        get_reward: Callable[[List[SA]], List[float]],
        policy_function: PolicyFunction,
        state_list: List[RLStateI]) -> List[Tuple[SA, List[SA], float, List[float]]]:
    # y is not used
    # Sample y_s
    c_log.debug("monte_carlo_explore")
    pk = PromiseKeeper(get_reward)

    def get_reward_future(sa: Tuple[State, Action]) -> MyFuture:
        return MyPromise(sa, pk).future()

    batch_size = len(state_list)
    c_log.debug(" - Computing mean actions")
    y_hat: List[Action] = policy_function.get_mean_action(state_list)
    c_log.debug(" - Sampling actions")
    y_s_list: List[List[Action]] = policy_function.sample_actions(state_list)
    c_log.debug(" - building action candidates")
    e_list = []
    for i in range(batch_size):
        x = state_list[i]
        sa: Tuple[RLStateI, Action] = x, y_hat[i]
        sa_list: List[Tuple[RLStateI, Action]] = [(x, y_s) for y_s in y_s_list[i]]
        reward_future = get_reward_future(sa)
        rf_list = [get_reward_future(sa) for sa in sa_list]
        e = sa, sa_list, reward_future, rf_list
        e_list.append(e)
    c_log.debug(" - Running action candidates")
    pk.do_duty()
    c_log.debug("monte_carlo_explore DONE")

    def unpack_future(e: Tuple[SA, List[SA], MyFuture, List[MyFuture]]):
        sa, sa_list, reward_future, rf_list = e
        return sa, sa_list, reward_future.get(), list_future(rf_list)

    return list(map(unpack_future, e_list))
