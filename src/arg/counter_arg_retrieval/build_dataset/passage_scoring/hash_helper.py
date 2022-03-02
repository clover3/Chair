import hashlib
from typing import List, Iterable


def get_int_list_hash(i_list: List[int]):
    ba = bytearray()
    for v in i_list:
        b = v.to_bytes(4, 'big')
        ba.extend(b)

    hash_val = hashlib.md5(ba).digest()
    return hash_val


def get_int_list_tuples_hash(i_list_tuples: Iterable[List[int]]) -> bytes:
    ba = bytearray()
    for i_list in i_list_tuples:
        for v in i_list:
            b = v.to_bytes(4, 'big')
            ba.extend(b)

    hash_val = hashlib.md5(ba).digest()
    return hash_val


def main():
    source = [1, 232, 2922], [1, 232, 2922]
    v = get_int_list_tuples_hash(source)
    print(v)
    print(type(v))



if __name__ == "__main__":
    main()