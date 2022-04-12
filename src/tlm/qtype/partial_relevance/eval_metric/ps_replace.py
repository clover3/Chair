
from typing import List, Callable

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from data_generator.tokenizer_wo_tf import get_tokenizer
from alignment.data_structure.eval_data_structure import RelatedBinaryAnswer
from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from tlm.qtype.partial_relevance.eval_metric.ep_common import EvalMetricBinaryIF
from trainer.promise import MyPromise, PromiseKeeper, MyFuture, list_future


# Paired-span Replacement
# Replace paired parts of query and document
# For some new word w
# If F(q, d) then F(q-qt+w, d-dt+w)
# Take maximum of tested w

def discretized_average(scores):
    pos_indices: List[int] = [idx for idx, s in enumerate(scores) if s > 0.5]
    pos_rate = len(pos_indices) / len(scores)
    return pos_rate


def discretized_average_minus(scores):
    pos_rate = discretized_average(scores)
    return 1 - pos_rate


class PSReplace(EvalMetricBinaryIF):
    def __init__(self,
                 forward_fn,
                 pair_modify_fn: Callable[[RelatedBinaryAnswer, RelatedEvalInstance, List[int]], SegmentedInstance],
                 get_word_pool: Callable[[str], List[List[int]]],
                 score_combine_fn: Callable[[List[float]], float]
                 ):
        self.pk = PromiseKeeper(forward_fn, 0.035)
        self.tokenizer = get_tokenizer()
        self.pair_modify_fn = pair_modify_fn
        self.get_word_pool: Callable[[str], List[List[int]]] = get_word_pool
        self.score_combine_fn = score_combine_fn

    def seg_to_future(self, seg: SegmentedInstance) -> MyFuture:
        return MyPromise(seg, self.pk).future()

    def get_predictions_for_case(self,
                                 problem: RelatedEvalInstance,
                                 answer: RelatedBinaryAnswer,
                                 ):
        key = problem.problem_id
        word_pool: List[List[int]] = self.get_word_pool(key)
        if not word_pool:
            raise IndexError()
        future_list: List[MyFuture] = []
        for word in word_pool:
            seg: SegmentedInstance = self.pair_modify_fn(answer, problem, word)
            new_qd_future: MyFuture = self.seg_to_future(seg)
            future_list.append(new_qd_future)
        return future_list

    def convert_future_to_score(self, future_list) -> float:
        scores = list_future(future_list)
        pos_rate = self.score_combine_fn(scores)
        return pos_rate

    def do_duty(self):
        self.pk.do_duty()

    def reset(self):
        self.pk.reset()