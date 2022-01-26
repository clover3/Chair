
from typing import List, Tuple, Dict, Callable

from data_generator.tokenizer_wo_tf import JoinEncoder
from list_lib import left
from misc_lib import get_second
from tlm.qtype.partial_relevance.attention_based.attention_mask_eval import indices_to_mask_dict
from tlm.qtype.partial_relevance.attention_based.bert_mask_predictor import get_bert_mask_predictor
from tlm.qtype.partial_relevance.attention_based.bert_masking_common import BERTMaskIF, logits_to_score
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import ComplementSearchOutput
from tlm.qtype.partial_relevance.eval_data_structure import ContributionSummary
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, RelatedEvalAnswer, SegmentedInstance
from tlm.qtype.partial_relevance.eval_metric.ep_common import EvalMetricIF, TupleOfListFuture
from trainer.promise import MyPromise, PromiseKeeper


class EvalMetricByAttentionDrop(EvalMetricIF):
    def __init__(self,
                 forward_fn: Callable[[List[Tuple[SegmentedInstance, Dict]]], List[float]],
                 drop_k, preserve_seg_idx):
        self.pk = PromiseKeeper(forward_fn)
        self.preserve_seg_idx = preserve_seg_idx
        self.drop_k = drop_k

    def get_predictions_for_case(self,
                                 problem: RelatedEvalInstance,
                                 answer: RelatedEvalAnswer,
                                 complement: ComplementSearchOutput) -> TupleOfListFuture:
        def get_future(seg: SegmentedInstance, mask: Dict):
            item = seg, mask
            return MyPromise(item, self.pk).future()

        mask_d = get_mask(answer.contribution,
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
        before_score = before.get()
        after_score = after.get()
        eval_score = before_score - after_score
        print("{0:.2f} -> {1:.2f}".format(before_score, after_score))
        return eval_score

    def do_duty(self):
        self.pk.do_duty(log_size=True)


class AttentionEvalCore:
    def __init__(self, client: BERTMaskIF, max_seq_length):
        self.client = client
        self.join_encoder = JoinEncoder(max_seq_length)

    def eval(self, items: List[Tuple[SegmentedInstance, Dict]]):
        payload = []
        for inst, mask_d in items:
            x0, x1, x2 = self.join_encoder.join(inst.text1.tokens_ids, inst.text2.tokens_ids)
            payload_item = x0, x1, x2, mask_d
            payload.append(payload_item)
        output = self.client.predict(payload)
        return [logits_to_score(e) for e in output]


def get_mask(contrib: ContributionSummary, drop_k: float, inst: SegmentedInstance, q_idx):
    l = []
    for d_idx in range(inst.text2.get_seg_len()):
        k = q_idx, d_idx
        v = contrib.table[q_idx][d_idx]
        l.append((k, v))
    l.sort(key=get_second, reverse=True)
    total_item = inst.text2.get_seg_len()
    keep_portion = 1 - drop_k
    n_item = int(total_item * keep_portion)
    drop_indices: List[Tuple[int, int]] = left(l[n_item:])
    print(inst.text1.get_tokens_for_seg(q_idx))
    print("Drop {} of {}".format(len(drop_indices), total_item))
    mask_d = indices_to_mask_dict(drop_indices)
    return mask_d


def get_attn_mask_forward_fn() -> Callable[[List[Tuple[SegmentedInstance, Dict]]], List[float]]:
    core = AttentionEvalCore(get_bert_mask_predictor(), max_seq_length=512)
    return core.eval
