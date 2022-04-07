

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from bert_api.segmented_instance.segmented_text import SegmentedText
from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import PartialSegment
from contradiction.alignment.data_structure.eval_data_structure import RelatedBinaryAnswer
from tlm.qtype.partial_relevance.related_eval_instance import RelatedEvalInstance
from tlm.qtype.partial_relevance.eval_metric.ep_common import TupleOfListFuture, EvalMetricBinaryIF
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import DocModFuncB
from trainer.promise import MyPromise, PromiseKeeper, MyFuture, list_future


# Paired-span Deletion
# Delete paired parts of query and document
# If F(q, d) then F(q-qt, d-dt)
class PSDeletion(EvalMetricBinaryIF):
    def __init__(self,
                 forward_fn,
                 seg_join_policy,
                 target_seg_idx,
                 drop_seg_idx,
                 doc_modify_fn: DocModFuncB,
                 ):
        self.seg_join_policy = seg_join_policy
        self.pk = PromiseKeeper(forward_fn, 0.035)
        self.target_seg_idx = target_seg_idx
        self.drop_seg_idx = drop_seg_idx
        self.tokenizer = get_tokenizer()
        self.doc_modify_fn: DocModFuncB = doc_modify_fn

    def seg_to_future(self, seg: SegmentedInstance) -> MyFuture:
        return MyPromise(seg, self.pk).future()

    def drop_core(self, text: SegmentedText, drop_seg_idx) -> SegmentedText:
        complement = PartialSegment(([]), 1)
        if drop_seg_idx == 1:
            keep_seg_idx = 0
        elif drop_seg_idx == 0:
            keep_seg_idx = 1
        else:
            raise ValueError

        return self.seg_join_policy.join_tokens(text, complement, keep_seg_idx)

    def get_predictions_for_case(self,
                                 problem: RelatedEvalInstance,
                                 answer: RelatedBinaryAnswer,
                                 ) -> TupleOfListFuture:
        def get_future(text1, text2):
            return self.seg_to_future(SegmentedInstance(text1, text2))

        full_query = problem.seg_instance.text1
        drop_doc = self.doc_modify_fn(problem.seg_instance.text2,
                                      answer.score_table[self.drop_seg_idx],
                                      )

        qd = get_future(full_query, problem.seg_instance.text2)
        before_futures = [qd]
        drop_query: SegmentedText = self.drop_core(problem.seg_instance.text1, self.drop_seg_idx)
        drop_q_drop_d = get_future(drop_query, drop_doc)
        after_futures = [drop_q_drop_d]

        future_predictions = before_futures, after_futures
        return future_predictions

    def convert_future_to_score(self, future_prediction_list) -> float:
        before, after = future_prediction_list
        qd, = list_future(before)
        drop_q_drop_d, = list_future(after)
        if qd < 0.5:
            print("WARNING the score for original doc,query should be relevant, but got {}".format(qd))
        score = drop_q_drop_d / qd
        return score

    def do_duty(self):
        self.pk.do_duty(log_size=True)

