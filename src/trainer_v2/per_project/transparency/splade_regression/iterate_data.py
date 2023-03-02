from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator
import sys


def iterate_triplet(data_path) -> Iterator[Tuple[str, str, str]]:
    f = open(data_path, "r", encoding="utf-8")
    for line in f:
        query, d1, d2, s1, s2 = line.split("\t")
        yield query, d1, d2


def main():
    for item in iterate_triplet(sys.argv[1]):
        print(item)
        break
    return NotImplemented


if __name__ == "__main__":
    main()