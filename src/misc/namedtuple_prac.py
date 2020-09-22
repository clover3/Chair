from typing import NamedTuple


class Triple(NamedTuple):
    one: int
    two: str
    three: str





l = [1, "2", 3]

triple = Triple(*l)
print(triple)