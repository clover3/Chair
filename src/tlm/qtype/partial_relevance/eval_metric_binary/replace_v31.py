from typing import List, Optional, Tuple, Dict

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from bert_api.segmented_instance.segmented_text import SegmentedText
from misc_lib import average
from contradiction.alignment.data_structure.eval_data_structure import RelatedEvalInstanceEx, RelatedBinaryAnswer
from tlm.qtype.partial_relevance.eval_metric.ep_common import ReplaceSamplePolicyIF
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import get_partial_text_as_segment
from tlm.qtype.partial_relevance.eval_metric_binary.eval_v3_common import EvalV3StateIF, V3StateWorkerIF, MetricV3, \
    FinalStateWorker
from tlm.qtype.partial_relevance.eval_metric_binary.replace_v2 import PerWord, add_w
from tlm.qtype.partial_relevance.eval_metric_binary.replace_v3 import ReplaceV3StateDone, \
    ReplaceV3StateWorkerBegin, ReplaceV3StateBegin
from trainer.promise import MyFuture, MyPromise, list_future, PromiseKeeper

EvalV31StateIF = EvalV3StateIF
ReplaceV31StateBegin = ReplaceV3StateBegin
ReplaceV31StateDone = ReplaceV3StateDone


class ReplaceV31StateSearch(EvalV31StateIF):
    def __init__(self,
                 problem: RelatedEvalInstanceEx,
                 answer: RelatedBinaryAnswer,
                 word_list: List[PerWord],
                 found_word_list: List[PerWord],
                 ):
        self.problem = problem
        self.answer = answer
        self.word_list = word_list
        self.found_word_list = found_word_list

    def get_code(self):
        return 1

    def get_pw(self) -> PerWord:
        return self.word_list[0]


class ReplaceV31StateFound(EvalV31StateIF):
    def __init__(self,
                 problem: RelatedEvalInstanceEx,
                 answer: RelatedBinaryAnswer,
                 found_word_list: List[PerWord],
                 ):
        self.problem = problem
        self.answer = answer
        self.found_word_list = found_word_list

    def get_code(self):
        return 2


class ReplaceV31StateWorkerSearch(V3StateWorkerIF):
    # 1 -> 2
    def __init__(self, pk):
        self.pk = pk

    def map(self, item: ReplaceV31StateSearch) -> Optional[Tuple[MyFuture[float], MyFuture[float]]]:
        try:
            pw = item.get_pw()
            score_qt_w_dt_w_future = MyPromise(pw.qt_w_dt_w, self.pk).future()
            score_qt_w_w_future = MyPromise(pw.qt_w_w, self.pk).future()
            future_pair = score_qt_w_dt_w_future, score_qt_w_w_future
            assert item.get_code() == 1
            return future_pair
        except IndexError:
            return None

    def reduce(self, future_pair: Optional[Tuple[MyFuture[float], MyFuture[float]]],
               item: ReplaceV31StateSearch) -> EvalV31StateIF:
        if future_pair is None:
            return ReplaceV31StateFound(item.problem, item.answer, item.found_word_list)
        else:
            score_qt_w_dt_w_future, score_qt_w_w_future = future_pair
            score_qt_w_dt_w = score_qt_w_dt_w_future.get()
            score_qt_w_w = score_qt_w_w_future.get()
            f1 = score_qt_w_dt_w >= 0.5
            f2 = score_qt_w_w >= 0.5
            satisfy: bool = f1 and not f2
            if satisfy:
                new_list = item.found_word_list + [item.get_pw()]
            else:
                new_list = item.found_word_list
            return ReplaceV31StateSearch(item.problem, item.answer, item.word_list[1:], new_list)


class ReplaceV31StateWorkerBegin(ReplaceV3StateWorkerBegin):
    def reduce(self, pw_list: MyFuture[List[PerWord]], item: ReplaceV3StateBegin) -> EvalV3StateIF:
        return ReplaceV31StateSearch(item.problem, item.answer, pw_list.get(), [])


class ReplaceV31StateWorkerEval(V3StateWorkerIF):
    # 2 -> 3
    def __init__(self, pk, replace_sample_policy):
        self.pk = pk
        self.replace_sample_policy = replace_sample_policy

    def map(self, item: ReplaceV31StateFound) -> List[List[MyFuture[float]]]:
        problem = item.problem
        answer = item.answer
        fp_list = []
        for pw in item.found_word_list:
            qt = get_partial_text_as_segment(problem.seg_instance.text1, problem.target_seg_idx)
            qt_w: SegmentedText = add_w(qt, pw.word)
            drop_doc_list: List[SegmentedText] = self.replace_sample_policy.get_replaced_docs(
                problem.seg_instance.text2,
                answer.score_table[problem.target_seg_idx],
                pw.word
            )
            si_list = [SegmentedInstance(qt_w, doc) for doc in drop_doc_list]
            future_predictions = [MyPromise(si, self.pk).future() for si in si_list]
            fp_list.append(future_predictions)
        return fp_list

    def reduce(self, fp_list: List[List[MyFuture[float]]],
               item: ReplaceV31StateFound) -> ReplaceV31StateDone:

        score_per_w_list = []
        for fp in fp_list:
            scores = list_future(fp)
            score_per_w = self.replace_sample_policy.combine_results(scores)
            score_per_w_list.append(score_per_w)

        if score_per_w_list:
            final_score = average(score_per_w_list)
            assert final_score <= 1
        else:
            final_score = 1

        return ReplaceV31StateDone(item.problem, item.answer, final_score)


class ReplaceV31(MetricV3):
    def __init__(self, forward_fn,
                 replace_sample_policy: ReplaceSamplePolicyIF,
                 get_word_pool):
        pk = PromiseKeeper(forward_fn)
        self.pk = pk
        self.workers: Dict[int, V3StateWorkerIF] = {
            0: ReplaceV31StateWorkerBegin(get_word_pool),
            1: ReplaceV31StateWorkerSearch(pk),
            2: ReplaceV31StateWorkerEval(pk, replace_sample_policy),
            3: FinalStateWorker(3)
        }

    def get_state_worker(self, state_code):
        return self.workers[state_code]

    def get_first_state(self, p: RelatedEvalInstanceEx, a: RelatedBinaryAnswer) -> EvalV31StateIF:
        return ReplaceV31StateBegin(p, a)

    def do_duty(self):
        self.pk.do_duty(False, True)

    def is_final(self, code):
        return code == 3
