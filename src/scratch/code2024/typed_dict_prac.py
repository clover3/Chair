from dataclasses import dataclass
from typing import TypedDict

raw_dict = {
    'int_value': 10,
    'str_value': 'string'
}

raw_dict2 = {
    'int_value': 10,
}

raw_dict3 = {
    'int_value': 10,
    'str_value': 'string',
    'float_value': 0.1,
}

wrong_dict = {
    'int_value': 'string',
    'str_value': 10,
}


class IntAndStr(TypedDict):
    int_value: int
    str_value: str


class IntAndStrAndFloat(TypedDict):
    int_value: int
    str_value: str
    float_value: float


obj1: IntAndStr = raw_dict
obj2: IntAndStr = raw_dict2
obj3: IntAndStr = raw_dict3
obj4: IntAndStr = wrong_dict


v = obj1.int_value

print(obj1)

