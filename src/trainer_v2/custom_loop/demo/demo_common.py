from typing import List, Iterable, Callable, Tuple

import numpy as np

from contradiction.medical_claims.token_tagging.problem_loader import AlamriProblem, load_alamri_problem
from data_generator.tokenizer_wo_tf import pretty_tokens
from dataset_specific.mnli.mnli_reader import NLIPairData
from list_lib import MaxKeyValue
from misc_lib import two_digit_float
from trainer_v2.custom_loop.per_task.nli_ts_util import enum_hypo_token_tuple_from_tokens, EncodedSegmentIF


def iterate_and_demo(
        itr: Iterable[NLIPairData],
        enum_items_by_hypo_seg_enum: Callable[[NLIPairData], Iterable[EncodedSegmentIF]],
        predictor):
    for e in itr:
        print("prem: ", e.premise)
        most_neutral = MaxKeyValue()
        most_contradiction = MaxKeyValue()
        for input_item in enum_items_by_hypo_seg_enum(e):
            input_item: EncodedSegmentIF = input_item
            l_decision, g_decision = predictor(input_item.get_input(), training=False)
            g_decision = g_decision[0]
            l_decision = l_decision[0]
            def format_prob(probs):
                return ", ".join(map(two_digit_float, probs))

            # g_decision_s = format_prob(g_decision)
            g_pred = np.argmax(g_decision)
            l_pred = np.argmax(l_decision, axis=1)
            print(" Pred: {} ({})".format(g_pred, g_decision),  " label :", e.get_label_as_int())
            h_first = input_item.get_hypothesis_tokens(0)
            h_second = input_item.get_hypothesis_tokens(1)
            print(" hypo1 ({}): {}".format(format_prob(l_decision[0]), pretty_tokens(h_first, True)))
            print(" hypo2 ({}): {}".format(format_prob(l_decision[1]), pretty_tokens(h_second, True)))

            most_neutral.update(h_second, l_decision[1][1])
            most_contradiction.update(h_second, l_decision[1][2])
        print("most_neutral:", pretty_tokens(most_neutral.max_key), most_neutral.max_value)
        print("most_contradiction:", pretty_tokens(most_contradiction.max_key), most_contradiction.max_value)
        input("Press enter to continue")


class EncodedSegment(EncodedSegmentIF):
    def __init__(self, input_x, p_tokens, h_tokens_list, st, ed):
        self.input_x = input_x
        self.h_tokens_list = h_tokens_list
        self.p_tokens = p_tokens
        self.st = st
        self.ed = ed

    def get_input(self):
        return self.input_x

    def get_premise_tokens(self):
        return self.p_tokens

    def get_hypothesis_tokens(self, segment_idx):
        return self.h_tokens_list[segment_idx]


def enum_hypo_token_tuple(tokenizer, hypothesis, window_size) -> List[Tuple[List[str], List[str], int, int]]:
    space_tokenized_tokens = hypothesis.split()
    yield from enum_hypo_token_tuple_from_tokens(tokenizer, space_tokenized_tokens, window_size)


def iter_alamri() -> Iterable[NLIPairData]:
    problems: List[AlamriProblem] = load_alamri_problem()

    for p in problems:
        yield NLIPairData(p.text1, p.text2, "neutral", "")