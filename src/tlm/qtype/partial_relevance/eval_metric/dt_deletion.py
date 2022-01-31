from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, RelatedEvalAnswer, SegmentedInstance
from tlm.qtype.partial_relevance.eval_metric.ep_common import TupleOfListFuture, EvalMetricIF, \
    EvalMetricWCBuilder
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import DocModFunc
from trainer.promise import MyPromise, PromiseKeeper, MyFuture, list_future


class DTDeletion(EvalMetricIF):
    def __init__(self, forward_fn, seg_join_policy, target_seg_idx, doc_modify_fn: DocModFunc):
        self.seg_join_policy = seg_join_policy
        self.pk = PromiseKeeper(forward_fn, 0.035)
        self.target_seg_idx = target_seg_idx
        self.tokenizer = get_tokenizer()
        self.doc_modify_fn: DocModFunc = doc_modify_fn

    def seg_to_future(self, seg: SegmentedInstance) -> MyFuture:
        return MyPromise(seg, self.pk).future()

    def get_predictions_for_case(self,
                                 problem: RelatedEvalInstance,
                                 answer: RelatedEvalAnswer,
                                 ) -> TupleOfListFuture:
        def get_future(text1, text2):
            return self.seg_to_future(SegmentedInstance(text1, text2))

        full_query = problem.seg_instance.text1

        drop_doc = self.doc_modify_fn(problem.seg_instance.text2,
                                      answer.contribution.table[self.target_seg_idx],
                                      )

        qd = get_future(full_query, problem.seg_instance.text2)
        before_futures = [qd]

        drop_d = get_future(full_query, drop_doc)
        after_futures = [drop_d]

        future_predictions = before_futures, after_futures
        return future_predictions

    def convert_future_to_score(self, future_prediction_list) -> float:
        before, after = future_prediction_list
        qd, = list_future(before)
        drop_d, = list_future(after)
        change_w_full_q = qd - drop_d
        return change_w_full_q

    def do_duty(self):
        self.pk.do_duty(log_size=True)


class EvalMetricLeaveOne(EvalMetricIF):
    def __init__(self, forward_fn, seg_join_policy, target_seg_idx, doc_modify_fn: DocModFunc):
        self.seg_join_policy = seg_join_policy
        self.pk = PromiseKeeper(forward_fn, 0.035)
        self.target_seg_idx = target_seg_idx
        self.tokenizer = get_tokenizer()
        self.doc_modify_fn: DocModFunc = doc_modify_fn

    def seg_to_future(self, seg: SegmentedInstance) -> MyFuture:
        return MyPromise(seg, self.pk).future()

    def get_predictions_for_case(self,
                                 problem: RelatedEvalInstance,
                                 answer: RelatedEvalAnswer,
                                 ) -> TupleOfListFuture:
        def get_future(text1, text2):
            return self.seg_to_future(SegmentedInstance(text1, text2))

        full_query = problem.seg_instance.text1

        drop_doc = self.doc_modify_fn(problem.seg_instance.text2,
                                      answer.contribution.table[self.target_seg_idx],
                                      )

        qd = get_future(full_query, problem.seg_instance.text2)
        before_futures = [qd]

        drop_d = get_future(full_query, drop_doc)
        after_futures = [drop_d]

        future_predictions = before_futures, after_futures
        return future_predictions

    def convert_future_to_score(self, future_prediction_list) -> float:
        before, after = future_prediction_list
        qd, = list_future(before)
        drop_d, = list_future(after)
        return min(qd, drop_d)

    def do_duty(self):
        self.pk.do_duty(log_size=True)


class EvalMetricByErasureNoSegWC(EvalMetricWCBuilder):
    def __init__(self, forward_fn, seg_join_policy, target_seg_idx, doc_modify_fn: DocModFunc):
        inner = DTDeletion(forward_fn, seg_join_policy, target_seg_idx, doc_modify_fn)
        super(EvalMetricByErasureNoSegWC, self).__init__(inner)


class EvalMetricLeaveOneWC(EvalMetricWCBuilder):
    def __init__(self, forward_fn, seg_join_policy, target_seg_idx, doc_modify_fn: DocModFunc):
        inner = EvalMetricLeaveOne(forward_fn, seg_join_policy, target_seg_idx, doc_modify_fn)
        super(EvalMetricLeaveOneWC, self).__init__(inner)


