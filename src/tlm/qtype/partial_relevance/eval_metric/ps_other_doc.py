
from typing import List, Dict, Tuple

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from data_generator.tokenizer_wo_tf import get_tokenizer
from alignment.data_structure.eval_data_structure import RelatedEvalAnswer
from alignment.data_structure.related_eval_instance import RelatedEvalInstance
from tlm.qtype.partial_relevance.eval_metric.ep_common import EvalMetricIF
from trainer.promise import MyPromise, PromiseKeeper, MyFuture, list_future


# Other documents

# Preprocessing
# For the aligned (qt,dt),
# Search dt's occurrence in the collection*
#    For each document, search complement for (qt, d, dt) (by span_iter)
#    If complement are found,
#       save data where,
#       data: List[Tuple[Document, List[Complement, score]]

#   Eval
#   For each document, get max of scores as document score
#   Return average document scores
#   Alternative, return count


class PSOtherDoc(EvalMetricIF):
    def __init__(self,
                 forward_fn,
                 target_seg_idx,
                 drop_seg_idx,
                 searched_doc_d: Dict[str, List[Tuple]],
                 ):
        self.pk = PromiseKeeper(forward_fn, 0.035)
        self.target_seg_idx = target_seg_idx
        self.drop_seg_idx = drop_seg_idx
        self.tokenizer = get_tokenizer()
        self.searched_doc_d = searched_doc_d

    def seg_to_future(self, seg: SegmentedInstance) -> MyFuture:
        return MyPromise(seg, self.pk).future()

    def get_predictions_for_case(self,
                                 problem: RelatedEvalInstance,
                                 answer: RelatedEvalAnswer,
                                 ):
        def get_future(text1, text2):
            return self.seg_to_future(SegmentedInstance(text1, text2))

        full_query = problem.seg_instance.text1

        key = problem.problem_id, self.drop_seg_idx
        word_pool: List[List[int]] = self.word_pool[key]
        if not word_pool:
            raise IndexError()
        future_list = []
        for word in word_pool:
            new_query = self.query_modify_fn(full_query, self.drop_seg_idx, word)
            new_doc = self.doc_modify_fn(problem.seg_instance.text2,
                                         answer.contribution.table[self.drop_seg_idx], word)
            new_qd_future = get_future(new_query, new_doc)
            future_list.append(new_qd_future)
        return future_list

    def convert_future_to_score(self, future_list) -> float:
        scores = list_future(future_list)
        score = max(scores)
        return score

    def do_duty(self):
        self.pk.do_duty(log_size=True)

