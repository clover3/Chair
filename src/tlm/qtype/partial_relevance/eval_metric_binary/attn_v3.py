from typing import Dict, Tuple

from tlm.qtype.partial_relevance.bert_mask_interface.mmd_z_mask_cacche import AttnMaskForward
from tlm.qtype.partial_relevance.eval_data_structure import SegmentedInstance
from tlm.qtype.partial_relevance.eval_metric.attn_mask_utils import get_drop_mask_binary
from tlm.qtype.partial_relevance.eval_metric.eval_by_attn import paired_future_diff
from tlm.qtype.partial_relevance.eval_metric_binary.eval_v3_common import assert_code
from tlm.qtype.partial_relevance.eval_metric_binary.one_step_common import OneStepStateBegin, OneStepStateDone, \
    SingleStateEvalMetric, SingleStateWorkerIF
from trainer.promise import MyFuture, MyPromise, PromiseKeeper


class AttnStateWorkerBegin(SingleStateWorkerIF):
    # 0 -> 1
    def __init__(self, forward_fn: AttnMaskForward):
        self.pk = PromiseKeeper(forward_fn)

    def map(self, item: OneStepStateBegin):
        assert_code(item.get_code(), 0)
        problem = item.problem
        answer = item.answer

        def get_future(seg: SegmentedInstance, mask: Dict):
            item = seg, mask
            return MyPromise(item, self.pk).future()

        mask_d = get_drop_mask_binary(answer.score_table, problem.seg_instance,
                                      problem.target_seg_idx)
        # Without attention mask drop
        before_futures = get_future(problem.seg_instance, {})
        after_futures = get_future(problem.seg_instance, mask_d)
        # With attention drop
        future_predictions = before_futures, after_futures
        return future_predictions

    def reduce(self, tuple: Tuple[MyFuture[float], MyFuture[float]], item: OneStepStateBegin)\
            -> OneStepStateDone:
        score = paired_future_diff(tuple)
        return OneStepStateDone(item.problem, item.answer, score)

    def do_duty(self):
        self.pk.do_duty()


# Inheritance hierarchy
# V3StateWorkerIF
# <- MetricV3 : (defines apply_map, apply_reduce)
# <- SingleStateEvalMetric (defines get_first_state, get_state_worker, get_scores)
# <- AttnEvalMetric

class AttnEvalMetric(SingleStateEvalMetric):
    def __init__(self, forward_fn: AttnMaskForward):
        begin_worker = AttnStateWorkerBegin(forward_fn)
        super(AttnEvalMetric, self).__init__(begin_worker)

