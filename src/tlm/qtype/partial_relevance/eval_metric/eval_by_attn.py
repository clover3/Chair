
from typing import List, Tuple

from data_generator.tokenizer_wo_tf import JoinEncoder
from list_lib import left
from misc_lib import get_second
from tlm.qtype.partial_relevance.attention_based.attention_mask_eval import indices_to_mask_dict
from tlm.qtype.partial_relevance.attention_based.bert_masking_common import BERTMaskIF, logits_to_score
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import ComplementSearchOutput
from tlm.qtype.partial_relevance.eval_data_structure import ContributionSummary
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, RelatedEvalAnswer, SegmentedInstance
from tlm.qtype.partial_relevance.eval_metric.doc_modify_fns import DocModFunc
from tlm.qtype.partial_relevance.eval_metric.ep_common import EvalMetricIF, TupleOfListFuture


class EvalMetricByAttentionDrop(EvalMetricIF):
    def __init__(self, forward_fn, seg_join_policy, preserve_seg_idx, doc_modify_fn: DocModFunc):
        pass

    def get_predictions_for_case(self,
                                 problem: RelatedEvalInstance,
                                 answer: RelatedEvalAnswer,
                                 complement: ComplementSearchOutput) -> TupleOfListFuture:
        pass

    def convert_future_to_score(self, future_prediction_list) -> float:
        pass

    def do_duty(self):
        pass


class AttentionEvalCore:
    def __init__(self, client: BERTMaskIF, max_seq_length, drop_k):
        self.client = client
        self.max_seq_length = max_seq_length
        self.join_encoder = JoinEncoder(max_seq_length)
        self.drop_k = drop_k

    def eval(self, inst: SegmentedInstance, contrib: ContributionSummary, q_idx) -> float:
        x0, x1, x2 = self.join_encoder.join(inst.text1.tokens_ids, inst.text2.tokens_ids)
        l = []
        for d_idx in range(inst.text2.get_seg_len()):
            k = q_idx, d_idx
            v = contrib.table[q_idx][d_idx]
            l.append((k, v))
        l.sort(key=get_second, reverse=True)
        total_item = inst.text2.get_seg_len()
        payload = []
        keep_portion = 1 - self.drop_k
        n_item = int(total_item * keep_portion)
        drop_indices: List[Tuple[int, int]] = left(l[n_item:])
        mask_d = indices_to_mask_dict(drop_indices)
        payload_item = x0, x1, x2, mask_d
        payload.append(payload_item)

        base_item = x0, x1, x2, {}
        payload_ex = [base_item] + payload
        assert len(payload_ex) == 2
        output = self.client.predict(payload_ex)

        points = output[1:]
        scores: List[float] = [logits_to_score(p) for p in points]
        return scores[0]