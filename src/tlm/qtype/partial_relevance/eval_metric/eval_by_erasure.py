from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import ComplementSearchOutput, PartialSegment
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, SegmentedInstance, \
    RelatedBinaryAnswer
from tlm.qtype.partial_relevance.eval_metric.ep_common import TupleOfListFuture, EvalMetricBinaryWCIF
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import DocModFuncB
from tlm.qtype.partial_relevance.segmented_text import SegmentedText
from trainer.promise import MyPromise, PromiseKeeper, MyFuture, list_future


class EvalMetricByErasure(EvalMetricBinaryWCIF):
    def __init__(self, forward_fn, seg_join_policy, preserve_seg_idx, doc_modify_fn: DocModFuncB):
        self.seg_join_policy = seg_join_policy
        self.pk = PromiseKeeper(forward_fn, 0.035)
        self.preserve_seg_idx = preserve_seg_idx
        self.tokenizer = get_tokenizer()
        self.doc_modify_fn: DocModFuncB = doc_modify_fn

    def drop_core(self, text: SegmentedText) -> SegmentedText:
        complement = PartialSegment(([]), 1)
        if self.preserve_seg_idx == 1:
            keep_seg_idx = 0
        elif self.preserve_seg_idx == 0:
            keep_seg_idx = 1
        else:
            raise ValueError

        return self.seg_join_policy.join_tokens(text, complement, keep_seg_idx)

    def seg_to_future(self, seg: SegmentedInstance) -> MyFuture:
        return MyPromise(seg, self.pk).future()

    def get_predictions_for_case(self,
                                 problem: RelatedEvalInstance,
                                 answer: RelatedBinaryAnswer,
                                 complement: ComplementSearchOutput) -> TupleOfListFuture:
        def get_future(text1, text2):
            return self.seg_to_future(SegmentedInstance(text1, text2))

        full_query = problem.seg_instance.text1

        drop_query: SegmentedText = self.drop_core(problem.seg_instance.text1)
        drop_doc = self.doc_modify_fn(problem.seg_instance.text2, answer.score_table[self.preserve_seg_idx])

        qd = get_future(full_query, problem.seg_instance.text2)
        drop_q_d = get_future(drop_query, problem.seg_instance.text2)
        before_futures = [qd, drop_q_d]

        drop_d = get_future(full_query, drop_doc)
        drop_q_drop_d = get_future(drop_query, drop_doc)
        after_futures = [drop_d, drop_q_drop_d]

        future_predictions = before_futures, after_futures
        return future_predictions

    def convert_future_to_score(self, future_prediction_list) -> float:
        before, after = future_prediction_list
        qd, drop_q = list_future(before)
        drop_d, drop_q_drop_d = list_future(after)
        change_w_full_q = qd - drop_d
        change_w_drop_q = drop_q-drop_q_drop_d
        eval_score = change_w_full_q - change_w_drop_q
        return eval_score

    def do_duty(self):
        self.pk.do_duty(log_size=True)

