from typing import List, Dict

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from bert_api.segmented_instance.segmented_text import SegmentedText
from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.qtype.partial_relevance.complement_search_pckg.complement_header import PartialSegment
from tlm.qtype.partial_relevance.complement_search_pckg.span_iter import get_candidates
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstanceEx, RelatedBinaryAnswer
from tlm.qtype.partial_relevance.eval_metric.ep_common import ReplaceSamplePolicyIF
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import get_partial_text_as_segment, get_replace_zero
from tlm.qtype.partial_relevance.eval_metric_binary.eval_v3_common import EvalV3StateIF, V3StateWorkerIF, MetricV3, \
    FinalStateWorker
from tlm.qtype.partial_relevance.eval_metric_binary.replace_v2 import PerWord, add_w
from tlm.qtype.partial_relevance.eval_metric_binary.replace_v3 import ReplaceV3StateDone, \
    ReplaceV3StateWorkerSearch, ReplaceV3StateWorkerEval, ReplaceV3StateBegin
from trainer.promise import MyFuture, PromiseKeeper

EvalV32StateIF = EvalV3StateIF
ReplaceV32StateBegin = ReplaceV3StateBegin
ReplaceV32StateDone = ReplaceV3StateDone
ReplaceV32StateWorkerEval = ReplaceV3StateWorkerEval

class ReplaceV32StateSearch(EvalV3StateIF):
    def __init__(self,
                 problem: RelatedEvalInstanceEx,
                 answer: RelatedBinaryAnswer,
                 word_list: List[PerWord],
                 ):
        self.problem = problem
        self.answer = answer
        self.word_list = word_list

    def get_code(self):
        return 1

    def get_pw(self) -> PerWord:
        return self.word_list[0]


class ReplaceV32StateFound(EvalV3StateIF):
    def __init__(self,
                 problem: RelatedEvalInstanceEx,
                 answer: RelatedBinaryAnswer,
                 found_word: PerWord,
                 ):
        self.problem = problem
        self.answer = answer
        self.found_word = found_word

    def get_code(self):
        return 2


ReplaceV32StateWorkerSearch = ReplaceV3StateWorkerSearch


class ReplaceV32StateWorkerBegin(V3StateWorkerIF):
    # 0 -> 1
    def __init__(self,):
        self.replace_zero = get_replace_zero()
        self.tokenizer = get_tokenizer()
        self.ngram_list = list(range(1, 4))

    def get_candidates(self, problem: SegmentedText):
        return get_candidates(self.tokenizer, self.ngram_list, problem)

    def map(self, item: ReplaceV3StateBegin) -> MyFuture[List[PerWord]]:
        assert item.get_code() == 0
        problem = item.problem
        answer = item.answer
        segments_candidate_list: List[PartialSegment] = self.get_candidates(problem.seg_instance.text2)
        word_pool: List[List[int]] = [s.data for s in segments_candidate_list]
        if not word_pool:
            raise IndexError()
        pw_list: List[PerWord] = []
        qt = get_partial_text_as_segment(problem.seg_instance.text1, problem.target_seg_idx)
        for word in word_pool:
            qt_w = add_w(qt, word)
            dt_w = self.replace_zero(problem.seg_instance.text2, answer.score_table[problem.target_seg_idx], word)
            w = SegmentedText.from_tokens_ids(word)
            qt_w_dt_w: SegmentedInstance = SegmentedInstance(qt_w, dt_w)
            qt_w_w: SegmentedInstance = SegmentedInstance(qt_w, w)
            pw_list.append(PerWord(qt_w_dt_w, qt_w_w, word))

        future = MyFuture[List[PerWord]]()
        future.set_value(pw_list)
        return future

    def reduce(self, pw_list: MyFuture[List[PerWord]], item: ReplaceV3StateBegin) -> EvalV3StateIF:
        return ReplaceV32StateSearch(item.problem, item.answer, pw_list.get())


class ReplaceV32(MetricV3):
    def __init__(self, forward_fn,
                 replace_sample_policy: ReplaceSamplePolicyIF,
                 ):
        pk = PromiseKeeper(forward_fn)
        self.pk = pk
        self.workers: Dict[int, V3StateWorkerIF] = {
            0: ReplaceV32StateWorkerBegin(),
            1: ReplaceV32StateWorkerSearch(pk),
            2: ReplaceV32StateWorkerEval(pk, replace_sample_policy),
            3: FinalStateWorker(3)
        }

    def get_state_worker(self, state_code):
        return self.workers[state_code]

    def get_first_state(self, p: RelatedEvalInstanceEx, a: RelatedBinaryAnswer) -> EvalV32StateIF:
        return ReplaceV32StateBegin(p, a)

    def do_duty(self):
        self.pk.do_duty(False, True)

    def is_final(self, code):
        return code == 3
