import functools
from typing import List, NamedTuple, Tuple

from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance, SegmentedInstance, RelatedBinaryAnswer, \
    RelatedEvalInstanceEx
from tlm.qtype.partial_relevance.eval_metric.ep_common import ReplaceSamplePolicyIF
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import get_partial_text_as_segment, get_drop_zero, \
    get_replace_zero
from tlm.qtype.partial_relevance.segmented_text import SegmentedText
from trainer.promise import MyPromise, PromiseKeeper, MyFuture, list_future


def add_w(text: SegmentedText, tokens_ids):
    n_last = len(text.tokens_ids)
    n_add = len(tokens_ids)
    new_seg_indices = list(range(n_last, n_last+n_add))
    new_indices = text.seg_token_indices + [new_seg_indices]
    return SegmentedText(text.tokens_ids + tokens_ids, new_indices)


class PerWord(NamedTuple):
    qt_w_dt_w: SegmentedInstance
    qt_w_w: SegmentedInstance
    word: List[int]


tokenizer = get_tokenizer()

def per_word_pred(forward_fn, pw_list: List[PerWord]) -> Tuple[bool, PerWord]:
    matching_pw = None
    def pair_rep(seg_inst):
        s1 = pretty_tokens(tokenizer.convert_ids_to_tokens(seg_inst.text1.tokens_ids))
        s2 = pretty_tokens(tokenizer.convert_ids_to_tokens(seg_inst.text2.tokens_ids))
        return "({}, {})".format(s1, s2)

    for pw in pw_list:
        score_qt_w_dt_w, score_qt_w_w = forward_fn([pw.qt_w_dt_w, pw.qt_w_w])
        f1 = score_qt_w_dt_w >= 0.5
        f2 = score_qt_w_w >= 0.5
        # print("qt_w_dt_w", pair_rep(pw.qt_w_dt_w))
        # print("qt_w_w", pair_rep(pw.qt_w_w))
        # print(f1, f2)
        satisfy: bool = f1 and not f2
        if satisfy:
            matching_pw = pw
            break
    found = matching_pw is not None
    return found, matching_pw


class ReplaceV2SingleSeg:
    def __init__(self, forward_fn,
                 replace_sample_policy: ReplaceSamplePolicyIF,
                 target_seg_idx,
                 get_word_pool):
        self.pk = PromiseKeeper(forward_fn, 0.035)

        def pw_pk_fn(pw_ll: List[List[PerWord]]) -> List[Tuple[bool, PerWord]]:
            return [per_word_pred(forward_fn, l) for l in pw_ll]

        self.pw_pk = PromiseKeeper(pw_pk_fn)
        self.replace_sample_policy = replace_sample_policy
        self.target_seg_idx = target_seg_idx
        self.tokenizer = get_tokenizer()
        self.get_word_pool = get_word_pool
        self.drop_zero = get_drop_zero()
        self.replace_zero = get_replace_zero()

    def seg_to_future(self, seg: SegmentedInstance) -> MyFuture:
        return MyPromise(seg, self.pk).future()

    def pw_list_future(self, pw_list: List[PerWord]) -> MyFuture[Tuple[bool, PerWord]]:
        return MyPromise(pw_list, self.pw_pk).future()

    def get_future(self, text1, text2):
        return self.seg_to_future(SegmentedInstance(text1, text2))

    def get_condition_pf(self,
                         problem: RelatedEvalInstance,
                         answer: RelatedBinaryAnswer,
                         ) -> MyFuture[Tuple[bool, PerWord]]:
        key = problem.problem_id
        word_pool: List[List[int]] = self.get_word_pool(key)
        if not word_pool:
            raise IndexError()
        pw_list: List[PerWord] = []
        qt = get_partial_text_as_segment(problem.seg_instance.text1, self.target_seg_idx)
        for word in word_pool:
            qt_w = add_w(qt, word)
            dt_w = self.replace_zero(problem.seg_instance.text2, answer.score_table[self.target_seg_idx], word)
            w = SegmentedText.from_tokens_ids(word)
            qt_w_dt_w: SegmentedInstance = SegmentedInstance(qt_w, dt_w)
            qt_w_w: SegmentedInstance = SegmentedInstance(qt_w, w)
            pw_list.append(PerWord(qt_w_dt_w, qt_w_w, word))

        return self.pw_list_future(pw_list)

    def get_test_pf(self,
                    problem: RelatedEvalInstance,
                    answer: RelatedBinaryAnswer,
                    pw: PerWord,
                    ):
        qt = get_partial_text_as_segment(problem.seg_instance.text1, self.target_seg_idx)
        qt_w = add_w(qt, pw.word)
        drop_doc_list: List[SegmentedText] = self.replace_sample_policy.get_replaced_docs(
            problem.seg_instance.text2,
            answer.score_table[self.target_seg_idx],
            pw.word
        )
        get_future_fn = functools.partial(self.get_future, qt_w)
        future_predictions = list(map(get_future_fn, drop_doc_list))
        return future_predictions

    def convert_condition_pf(self, fw: MyFuture[Tuple[bool, PerWord]]) -> Tuple[bool, PerWord]:
        found, pw = fw.get()
        return found, pw

    def convert_test_pf(self, future_prediction_list) -> float:
        score_list = list_future(future_prediction_list)
        return self.replace_sample_policy.combine_results(score_list)

    def do_duty(self):
        self.pw_pk.do_duty(log_size=False, reset=True)
        self.pk.do_duty(log_size=False, reset=True)


class ReplaceV2:
    def __init__(self, forward_fn,
                 replace_sample_policy: ReplaceSamplePolicyIF,
                 get_word_pool):
        self.pk = PromiseKeeper(forward_fn, 0.035)

        def pw_pk_fn(pw_ll: List[List[PerWord]]) -> List[Tuple[bool, PerWord]]:
            return [per_word_pred(forward_fn, l) for l in pw_ll]

        self.pw_pk = PromiseKeeper(pw_pk_fn)
        self.replace_sample_policy = replace_sample_policy
        self.tokenizer = get_tokenizer()
        self.get_word_pool = get_word_pool
        self.drop_zero = get_drop_zero()
        self.replace_zero = get_replace_zero()

    def seg_to_future(self, seg: SegmentedInstance) -> MyFuture:
        return MyPromise(seg, self.pk).future()

    def pw_list_future(self, pw_list: List[PerWord]) -> MyFuture[Tuple[bool, PerWord]]:
        return MyPromise(pw_list, self.pw_pk).future()

    def get_future(self, text1, text2):
        return self.seg_to_future(SegmentedInstance(text1, text2))

    def get_condition_pf(self,
                         problem: RelatedEvalInstanceEx,
                         answer: RelatedBinaryAnswer,
                         ) -> MyFuture[Tuple[bool, PerWord]]:
        key = problem.problem_id
        word_pool: List[List[int]] = self.get_word_pool(key)
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

        return self.pw_list_future(pw_list)

    def get_test_pf(self,
                    problem: RelatedEvalInstanceEx,
                    answer: RelatedBinaryAnswer,
                    pw: PerWord,
                    ):
        qt = get_partial_text_as_segment(problem.seg_instance.text1, problem.target_seg_idx)
        qt_w = add_w(qt, pw.word)
        drop_doc_list: List[SegmentedText] = self.replace_sample_policy.get_replaced_docs(
            problem.seg_instance.text2,
            answer.score_table[problem.target_seg_idx],
            pw.word
        )
        get_future_fn = functools.partial(self.get_future, qt_w)
        future_predictions = list(map(get_future_fn, drop_doc_list))
        return future_predictions

    def convert_condition_pf(self, fw: MyFuture[Tuple[bool, PerWord]]) -> Tuple[bool, PerWord]:
        found, pw = fw.get()
        return found, pw

    def convert_test_pf(self, future_prediction_list) -> float:
        score_list = list_future(future_prediction_list)
        return self.replace_sample_policy.combine_results(score_list)

    def do_duty(self):
        self.pw_pk.do_duty(log_size=False, reset=True)
        self.pk.do_duty(log_size=False, reset=True)
