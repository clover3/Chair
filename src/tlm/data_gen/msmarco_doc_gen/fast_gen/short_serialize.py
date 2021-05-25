from typing import List, Iterable, Callable, Dict, Tuple, Set


def to_big_bytes(i):
    assert i < 65536
    b = i.to_bytes(2, byteorder='big', signed=False)
    return b


def from_big_bytes(b):
    return int.from_bytes(b, 'big')


def to_utf8_bytes(s):
    return bytes(s, encoding="utf-8")


def serialize_int_list(int_list: List[int]) -> bytearray:
    frame = bytearray()
    for i in int_list:
        frame.extend(to_big_bytes(i))
    return frame


def deserialize_int_list(ba: bytearray) -> List[int]:
    cursor = 0
    l = []
    while cursor < len(ba):
        i = int.from_bytes(ba[cursor: cursor+2], 'big')
        l.append(i)
        cursor += 2
    return l


def check1():
    l1 = [28182, 65530, 1281, 1582, 281, 0]
    bs = serialize_int_list(l1)
    l2 = deserialize_int_list(bs)
    print(l1)
    print(l2)
    assert l1 == l2

def main():

    pass


if __name__ == "__main__":
    main()