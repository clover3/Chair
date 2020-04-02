from functools import partial
from typing import Dict, NamedTuple, List, Callable


class MyNamedTuple(NamedTuple):
    doc_id: str
    st: int


def send_queries_passage() -> Dict[str, List[MyNamedTuple]]:
    d = {}
    l = []
    l.append(MyNamedTuple(doc_id="id", st=0))

    d['sstr'] = l
    return d


def multiply(x: int, y: int) -> int:
    return x * y


def multi2(y: int) -> int:
    return 2 * y


def put4(f: Callable[[int], int]) -> int:
    return f(4)


# create a new function that multiplies by 2
mul2: partial[[int], int] = partial(multiply, 2)
print(mul2(4))

print(put4(mul2))   # warning : Expected (int)->, got partial[Any, int] instead,
print(put4(multi2))

