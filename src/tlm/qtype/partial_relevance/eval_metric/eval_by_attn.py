import abc
import functools
from typing import List, Tuple, Dict, Callable, NamedTuple

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from misc_lib import average
from tlm.qtype.partial_relevance.attention_based.bert_masking_common import logits_to_score, dist_l2
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, RelatedEvalAnswer, RelatedBinaryAnswer
from tlm.qtype.partial_relevance.eval_metric.attn_mask_utils import get_drop_mask, get_drop_mask_binary
from tlm.qtype.partial_relevance.eval_metric.ep_common import EvalMetricWCIF, TupleOfListFuture, EvalMetricIF, \
    EvalMetricBinaryIF
from trainer.promise import MyPromise, PromiseKeeper, list_future, MyFuture


class EvalMetricByAttentionDrop(EvalMetricIF):
    def __init__(self,
                 forward_fn: Callable[[List[Tuple[SegmentedInstance, Dict]]], List[List[float]]],
                 drop_k, preserve_seg_idx):
        self.pk = PromiseKeeper(forward_fn, 0.035)
        self.preserve_seg_idx = preserve_seg_idx
        self.drop_k = drop_k

    def get_predictions_for_case(self,
                                 problem: RelatedEvalInstance,
                                 answer: RelatedEvalAnswer,
                                 ) -> TupleOfListFuture:
        def get_future(seg: SegmentedInstance, mask: Dict):
            item = seg, mask
            return MyPromise(item, self.pk).future()
        mask_d = get_drop_mask(answer.contribution,
                               self.drop_k,
                               problem.seg_instance,
                               self.preserve_seg_idx)

        # Without attention mask drop
        before_futures = get_future(problem.seg_instance, {})
        after_futures = get_future(problem.seg_instance, mask_d)
        # With attention drop
        future_predictions = before_futures, after_futures
        return future_predictions

    def convert_future_to_score(self, future_prediction_list) -> float:
        before, after = future_prediction_list
        before_score = logits_to_score(before.get())
        after_score = logits_to_score(after.get())
        eval_score = before_score - after_score
        return eval_score

    def do_duty(self):
        self.pk.do_duty(log_size=True)


def paired_future_diff(future_prediction_list: Tuple[MyFuture, MyFuture]):
    before, after = future_prediction_list
    before_score = logits_to_score(before.get())
    after_score = logits_to_score(after.get())
    eval_score = before_score - after_score
    return eval_score


class EvalMetricByAttentionDropB(EvalMetricBinaryIF):
    def __init__(self,
                 forward_fn: Callable[[List[Tuple[SegmentedInstance, Dict]]], List[List[float]]],
                 preserve_seg_idx):
        self.pk = PromiseKeeper(forward_fn, 0.035)
        self.target_seg_idx = preserve_seg_idx

    def get_predictions_for_case(self,
                                 problem: RelatedEvalInstance,
                                 answer: RelatedBinaryAnswer,
                                 ) -> TupleOfListFuture:
        def get_future(seg: SegmentedInstance, mask: Dict):
            item = seg, mask
            return MyPromise(item, self.pk).future()
        mask_d = get_drop_mask_binary(answer.score_table, problem.seg_instance, self.target_seg_idx)
        # Without attention mask drop
        before_futures = get_future(problem.seg_instance, {})
        after_futures = get_future(problem.seg_instance, mask_d)
        # With attention drop
        future_predictions = before_futures, after_futures
        return future_predictions

    def convert_future_to_score(self, future_prediction_list) -> float:
        eval_score = paired_future_diff(future_prediction_list)
        return eval_score

    def do_duty(self):
        self.pk.do_duty(log_size=True)


def get_num_drop_mask(mask_d, seg) -> int:
    n_drop = 0
    for k, v in mask_d.items():
        if v == 1:
            idx1, idx2 = k
            assert idx1 < seg.text1.get_seg_len()
            assert idx2 < seg.text2.get_seg_len()
            n_drop += 1
    return n_drop


def get_num_used_mask(seg: SegmentedInstance, mask_d: Dict[Tuple[int, int], int]) -> int:
    n_drop = get_num_drop_mask(mask_d, seg)
    total_items = seg.text1.get_seg_len() * seg.text2.get_seg_len()
    n_used_mask = total_items - n_drop
    return n_used_mask


def get_sparsity(seg: SegmentedInstance, mask_d: Dict[Tuple[int, int], int]) -> float:
    total_items = seg.text1.get_seg_len() * seg.text2.get_seg_len()
    n_used_mask = get_num_used_mask(seg, mask_d)
    return n_used_mask / total_items


class PIF(NamedTuple):
    base_future: MyFuture[List[float]]
    after_futures_list: List[MyFuture[List[float]]]
    brevity_scores: List[float]


# Base Instance vs Neighbor Instances
class EvalMetricByAttentionMultiPoint(EvalMetricIF):
    def __init__(self,
                 forward_fn: Callable[[List[Tuple[SegmentedInstance, Dict]]], List[List[float]]],
                 target_seg_idx):
        self.pk = PromiseKeeper(forward_fn, 0.035)
        self.target_seg_idx = target_seg_idx

    def get_predictions_for_case(self,
                                 problem: RelatedEvalInstance,
                                 answer: RelatedEvalAnswer,
                                 ) -> PIF:
        def get_future(seg: SegmentedInstance, mask: Dict) -> MyFuture[List[float]]:
            item = seg, mask
            return MyPromise(item, self.pk).future()

        keep_portion_list = [i / 10 for i in range(10)]
        drop_k_list = [1-v for v in keep_portion_list]

        def get_mask(drop_k):
            mask_d = get_drop_mask(answer.contribution,
                                   drop_k,
                                   problem.seg_instance,
                                   self.target_seg_idx)
            return mask_d

        mask_d_list: List[Dict] = list(map(get_mask, drop_k_list))
        get_sparsity_fn = functools.partial(get_sparsity, problem.seg_instance)
        sparsity_list: List[float] = list(map(get_sparsity_fn, mask_d_list))
        before_futures: MyFuture[List[float]] = get_future(problem.seg_instance, {})
        after_future_list: List[MyFuture[List[float]]] = [get_future(problem.seg_instance, m) for m in mask_d_list]
        future_predictions = PIF(before_futures, after_future_list, sparsity_list)
        return future_predictions

    @abc.abstractmethod
    def convert_future_to_score(self, f: PIF) -> float:
        pass

    def do_duty(self):
        self.pk.do_duty(log_size=True)


# Base Instance vs Neighbor Instances
class EvalMetricByAttentionAUC(EvalMetricByAttentionMultiPoint):
    def __init__(self,
                 forward_fn: Callable[[List[Tuple[SegmentedInstance, Dict]]], List[List[float]]],
                 target_seg_idx):
        super(EvalMetricByAttentionAUC, self).__init__(forward_fn, target_seg_idx)

    @abc.abstractmethod
    def convert_future_to_score(self, f: PIF) -> float:
        before_score = f.base_future.get()
        after_score_list = list_future(f.after_futures_list)
        error_list = [dist_l2(before_score, after_score) for after_score in after_score_list]
        return average(error_list)


class EvalMetricByAttentionDropWC(EvalMetricWCIF):
    def __init__(self, forward_fn: Callable[[List[Tuple[SegmentedInstance, Dict]]], List[List[float]]],
                 drop_k, preserve_seg_idx):
        inner = EvalMetricByAttentionDrop(forward_fn, drop_k, preserve_seg_idx)
        super(EvalMetricByAttentionDropWC, self).__init__(inner)


# Base Instance vs Neighbor Instances
class EvalMetricByAttentionBrevity(EvalMetricByAttentionMultiPoint):
    def __init__(self,
                 forward_fn: Callable[[List[Tuple[SegmentedInstance, Dict]]], List[List[float]]],
                 target_seg_idx):
        super(EvalMetricByAttentionBrevity, self).__init__(forward_fn, target_seg_idx)

    def convert_future_to_score(self, f: PIF) -> float:
        before_score: List[float] = f.base_future.get()
        after_score_list: List[List[float]] = list_future(f.after_futures_list)
        fidelity_loss_list: List[float] = [dist_l2(before_score, after_score) for after_score in after_score_list]

        def combine_score(s_pair: Tuple[float, float]):
            fidelity, brevity = s_pair
            return fidelity + brevity

        pair_list: List[Tuple[float, float]] = list(zip(fidelity_loss_list, f.brevity_scores))
        print("-------")
        for f, b in pair_list:
            print(f, b, combine_score((f, b)))
        return min(map(combine_score, pair_list))


class AttentionBrevityDetail:
    def __init__(self, eval_policy):
        self.inner = eval_policy

    def get_predictions_for_case(self,
                                 problem: RelatedEvalInstance,
                                 answer: RelatedEvalAnswer,
                                 ) -> PIF:
        return self.inner.get_predictions_for_case(problem, answer)

    def get_detail(self, f: PIF):
        before_score: List[float] = f.base_future.get()
        after_score_list: List[List[float]] = list_future(f.after_futures_list)
        fidelity_loss_list: List[float] = [dist_l2(before_score, after_score) for after_score in after_score_list]
        pair_list: List[Tuple[float, float]] = list(zip(fidelity_loss_list, f.brevity_scores))
        e_list = []
        for f, b in pair_list:
            e = {
                'fidelity': f,
                'brevity': b
            }
            e_list.append(e)
        return e_list