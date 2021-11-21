import csv
import datetime
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from typing import NamedTuple, NewType

import pytz

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


# Names should not include prefixes "Input." or "Answers."
class HITScheme(NamedTuple):
    inputs: List[ColumnName]
    answer_units: List[AnswerUnit]

    def get_answer_names(self):
        return list([ans.name for ans in self.answer_units])


class RadioButtonGroup(AnswerUnit):
    def __init__(self, prefix, post_fix, make_int=False):
        self.name = prefix
        self.pre_fix = prefix
        self.post_fix: List[str] = post_fix
        self.make_int = make_int

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
            if d[column_name].lower() == "true":
                assert answer is None
                answer = post
        if self.make_int:
            return int(answer) if answer is not None else None
        return answer


class YesNoRadioButtonGroup(AnswerUnit):
    def __init__(self, prefix):
        self.name = prefix
        self.pre_fix = prefix
        self.post_fix: List[str] = ['Yes', 'No']

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
            if d[column_name].lower() == "true":
                assert answer is None
                answer = post

        if answer == "Yes":
            return 1
        else:
            return 0


class Textbox(AnswerUnit):
    def __init__(self, name):
        self.name = ColumnName(name)

    def get_column_names(self) -> List[ColumnName]:
        return [self.name]

    def parse(self, d: Dict[ColumnName, str]):
        return d[self.name]


class Categorical(AnswerUnit):
    def __init__(self, column_name, options: Dict[str, Any]):
        self.name = column_name
        self.column_name = column_name
        self.options = options

    def get_column_names(self) -> List[ColumnName]:
        return [self.column_name]

    def parse(self, d: Dict[ColumnName, str]):
        answer_str = d[self.column_name]
        return self.options[answer_str]


class Checkbox(AnswerUnit):
    def __init__(self, column_name):
        self.name = column_name
        self.column_name = column_name

    def get_column_names(self) -> List[ColumnName]:
        return [self.column_name]

    def parse(self, d: Dict[ColumnName, str]):
        answer_str = d[self.column_name]
        return {"true": 1,
                "TRUE": 1,
                "FALSE": 0,
                "false": 0}[answer_str]


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
    def __init__(self, inputs_d, outputs_d, row):
        self.inputs: Dict[str, str] = inputs_d
        self.outputs: Dict = outputs_d
        self.assignment_id = row['AssignmentId']
        self.worker_id = row['WorkerId']
        self.hit_id = row['HITId']
        self.status = row['AssignmentStatus']
        self.work_time = row['WorkTimeInSeconds']
        self.accept_time: str = row["AcceptTime"]
        self.submit_time: str = row["SubmitTime"]

    def get_input(self, input_name: ColumnName):
        return self.inputs[input_name]

    def get_accept_time(self):
        return parse_mturk_time(self.accept_time)

    def get_submit_time(self):
        return parse_mturk_time(self.submit_time)

    def get_repeated_entries_result(self, name, idx):
        return self.outputs[name][idx]


def parse_file(path, hit_scheme: HITScheme, f_remove_rejected=True) -> List[HitResult]:
    f = open(path, "r", encoding="utf-8")
    data = []
    for row in csv.reader(f):
        data.append(row)
    head = list(data[0])

    def get_input_raw_column(column_name):
        return "Input." + column_name

    def get_output_raw_column(column_name):
        return "Answer." + column_name

    row_idx_d: Dict[str, int] = {}

    for idx, column_name in enumerate(head):
        row_idx_d[column_name] = idx

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
        all_value_d = {}
        for column_name in row_idx_d:
            row_idx = row_idx_d[column_name]
            try:
                all_value_d[column_name] = row[row_idx]
            except IndexError:
                pass

        for input_name in hit_scheme.inputs:
            inputs_d[input_name] = all_value_d[input_name]

        for answer_unit in hit_scheme.answer_units:
            unit_output = {}
            for column_name in answer_unit.get_column_names():
                unit_output[column_name] = all_value_d[column_name]
            answer_d[answer_unit.name] = answer_unit.parse(unit_output)
        return HitResult(inputs_d, answer_d, all_value_d)

    outputs = lmap(parse_row, data[1:])
    if f_remove_rejected:
        outputs = remove_rejected(outputs)
    return outputs


def remove_rejected(hit_results: List[HitResult]):
    output = []
    for h in hit_results:
        if h.status != "Rejected":
            output.append(h)
    return output


def parse_mturk_time(s):
    # return dateutil.parser.parse(s)
    tzs = "PDT"
    assert tzs in s
    s = s.replace(tzs + " ", "")
    # EDT = pytz.timezone('UTC-0400')
    PDT = pytz.timezone('US/Pacific')
    # "Sat Jul 03 23:28:53 PDT 2021"
    time_wo_tz = datetime.datetime.strptime(s, "%a %b %d %H:%M:%S %Y")
    time_w_tz = PDT.localize(time_wo_tz)
    # t = time_wo_tz
    # time_w_tz = datetime.datetime(t.year, t.month, t.day, t.hour, t.minute, t.second, tzinfo=PDT)
    return time_w_tz