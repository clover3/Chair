import csv
from abc import ABC, abstractmethod
from typing import List, Dict
from typing import NamedTuple, NewType

from list_lib import lmap

ColumnName = NewType('ColumnName', str)


class AnswerUnit(ABC):
    name: str

    @abstractmethod
    def get_column_names(self) -> List[ColumnName]:
        pass

    @abstractmethod
    def parse(self, d: Dict[ColumnName, str]):
        pass


class HITScheme(NamedTuple):
    inputs: List[ColumnName]
    answer_units: List[AnswerUnit]


class RadioButtonGroup(AnswerUnit):
    def __init__(self, prefix, post_fix):
        self.pre_fix = prefix
        self.post_fix: List[str] = post_fix

    def get_column_names(self) -> List[ColumnName]:
        l = []
        for post in self.post_fix:
            l.append(self.get_column_name_for_postfix(post))
        return l

    def get_column_name_for_postfix(self, post):
        return ColumnName(self.pre_fix + post + ".on")

    def parse(self, d: Dict[ColumnName, str]):
        answer = None
        for post in self.post_fix:
            column_name = self.get_column_name_for_postfix(post)
            if d[column_name] == "true":
                assert answer is None
                answer = post
        return answer


class RepeatedEntries(AnswerUnit):
    def __init__(self, name: str, prefix_list: List[str], postfix_list: List[str]):
        self.name = name
        self.sub_units: List[AnswerUnit] = []
        for prefix in prefix_list:
            radio_group = RadioButtonGroup(prefix, postfix_list)
            self.sub_units.append(radio_group)

    def get_column_names(self) -> List[ColumnName]:
        l = []
        for unit in self.sub_units:
            l.extend(unit.get_column_names())
        return l

    def parse(self, d: Dict[ColumnName, str]):
        r = []
        for unit in self.sub_units:
            r.append(unit.parse(d))
        return r


RadioGroupResult = NewType('RadioGroupResult', str)


class HitResult:
    def __init__(self, inputs_d, outputs_d):
        self.inputs: Dict[str, str] = inputs_d
        self.outputs: Dict = outputs_d

    def get_input(self, input_name: ColumnName):
        return self.inputs[input_name]

    def get_repeated_entries_result(self, name, idx):
        return self.outputs[name][idx]


def parse_file(path, hit_scheme: HITScheme) -> List[HitResult]:
    f = open(path, "r", encoding="utf-8")
    data = []
    for row in csv.reader(f):
        data.append(row)
    head = list(data[0])
    print(head)

    def get_input_raw_column(column_name):
        return "Input." + column_name

    def get_output_raw_column(column_name):
        return "Answer." + column_name

    row_idx_d: Dict[str, int] = {}
    for input_name in hit_scheme.inputs:
        idx = head.index(get_input_raw_column(input_name))
        row_idx_d[input_name] = idx

    for answer_unit in hit_scheme.answer_units:
        for column_name in answer_unit.get_column_names():
            idx = head.index(get_output_raw_column(column_name))
            row_idx_d[column_name] = idx

    def parse_row(row) -> HitResult:
        inputs_d = {}
        answer_d = {}
        for input_name in hit_scheme.inputs:
            row_idx = row_idx_d[input_name]
            value = row[row_idx]
            inputs_d[input_name] = value

        for answer_unit in hit_scheme.answer_units:
            unit_output = {}
            for column_name in answer_unit.get_column_names():
                row_idx = row_idx_d[column_name]
                value = row[row_idx]
                unit_output[column_name] = value
            answer_d[answer_unit.name] = answer_unit.parse(unit_output)
        return HitResult(inputs_d, answer_d)

    return lmap(parse_row, data[1:])
