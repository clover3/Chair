from typing import List

from list_lib import lmap
from misc_lib import SuccessCounter
from mturk.parse_util import ColumnName, Categorical, HITScheme, Checkbox, RadioButtonGroup, HitResult, parse_file, \
    YesNoRadioButtonGroup, Textbox


def get_ca_run1_scheme():
    inputs = [ColumnName("c_text"), ColumnName("p_text"), ColumnName("doc_id")]
    options = {
        "Not relevant": 0,
        "Relevant but not a counter-argument.": 1,
        "Counter-argument.": 2,
    }
    answer_units = [Categorical("relevant.label", options)]
    hit_scheme = HITScheme(inputs, answer_units)
    return hit_scheme


def get_ca4_scheme():
    inputs = [ColumnName("c_text"), ColumnName("p_text"), ColumnName("doc_id")]
    answer_units = []
    for i in range(1, 9):
        answer_units.append(Checkbox("Q{}.on".format(i)))
    answer_units.append(RadioButtonGroup("Q9.", lmap(str, range(4)), True))
    hit_scheme = HITScheme(inputs, answer_units)
    return hit_scheme


def get_ca_rate_common(input_path, hit_scheme, get_binary_answer_from_hit_result):
    hit_results: List[HitResult] = parse_file(input_path, hit_scheme)
    sc = SuccessCounter()
    for hit_result in hit_results:
        if get_binary_answer_from_hit_result(hit_result):
            sc.suc()
        else:
            sc.fail()
    return sc


def get_ca3_scheme():
    inputs = [ColumnName("c_text"), ColumnName("p_text"), ColumnName("doc_id")]
    q1 = YesNoRadioButtonGroup("Q1.")
    q2 = YesNoRadioButtonGroup("Q2.")
    q2_text = Textbox ("Q2")
    answer_units = [q1, q2, q2_text]
    hit_scheme = HITScheme(inputs, answer_units)
    return hit_scheme


def get_ca_run1_scheme2():
    inputs = [ColumnName("c_text"), ColumnName("p_text"), ColumnName("doc_id")]
    # answer_list = [
    #     "Q1.on",
    #     "Q14.0.Q14.0",
    #     "Q14.1.Q14.1",
    #     "Q14.2.Q14.2",
    #     "Q14.3.Q14.3",
    #     "Q2.on",
    #     "claim_arg_oppose.on",
    #     "claim_arg_support.on",
    #     "claim_info_oppose.on",
    #     "claim_info_support.on",
    #     "claim_mention.on",
    #     "related.on",
    #     "topic_arg_oppose.on",
    #     "topic_arg_support.on",
    #     "topic_info_oppose.on",
    #     "topic_info_support.on",
    #     "topic_mention.on",
    # ]
    answer_units = []
    for i in range(1, 14):
        answer_units.append(Checkbox("Q{}.on".format(i)))

    answer_units.append(RadioButtonGroup("Q14.", lmap(str, range(4)), True))
    hit_scheme = HITScheme(inputs, answer_units)
    return hit_scheme

