from typing import List

from list_lib import lmap
from mturk.parse_util import Checkbox, ColumnName, RadioButtonGroup, HITScheme, HitResult, parse_file


def get_ca2_2_scheme():
    answer_units = []
    for i in range(1, 14):
        answer_units.append(Checkbox("Q{}.on".format(i)))
    inputs = [ColumnName(name) for name in ["qid", "p_text", "c_text", "doc_id", "passage_idx", "passage"]]
    answer_units.append(RadioButtonGroup("Q14.", lmap(str, range(4)), True))
    hit_scheme = HITScheme(inputs, answer_units)
    return hit_scheme


class NickName:
    def __init__(self, source, nick_name_d):
        self.nick_name_d = nick_name_d
        self.source = source

    def __getitem__(self, key):
        if key in self.nick_name_d:
            return self.source[self.nick_name_d[key]]
        else:
            return self.source[key]

    def __setitem__(self, key, value):
        if key in self.nick_name_d:
            return self.source.__setitem__(self.nick_name_d[key], value)
        else:
            return self.source.__setitem__(key, value)

    def __str__(self):
        return self.source.__str__()

    def __len__(self):
        return self.source.__len__()


def add_nickname(hit: HitResult):
    nick_name_d = {
        "P_Support_C": "Q1.on",
        "P_Oppose_C": "Q2.on",
        "Related": "Q3.on",
        "Mention_C": "Q4.on",
        "Mention_P": "Q5.on",
        "Arg_Support_C": "Q6.on",
        "Arg_Oppose_C": "Q7.on",
        "Arg_Support_P": "Q8.on",
        "Arg_Oppose_P": "Q9.on",
        "Info_Support_C": "Q10.on",
        "Info_Oppose_C": "Q11.on",
        "Info_Support_P": "Q12.on",
        "Info_Oppose_P": "Q13.on",
        "Useful": "Q14.",
    }
    hit.outputs = NickName(hit.outputs, nick_name_d)
    nick_name_d = {
        'claim': "c_text",
        'perspective': "p_text"
    }
    hit.inputs = NickName(hit.inputs, nick_name_d)
    return hit


def load_file(csv_path) -> List[HitResult]:
    hit_scheme = get_ca2_2_scheme()
    hit_results: List[HitResult] = parse_file(csv_path, hit_scheme)
    return lmap(add_nickname, hit_results)