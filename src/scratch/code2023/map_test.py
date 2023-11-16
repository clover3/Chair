from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

from list_lib import lmap


def iterator_from(map_fn, int_iterable):
    for int_item in int_iterable:
        yield map_fn(int_item)



def main():

    range_items = range(10)

    map_items: Iterable[str] = map(str, range_items)
    lmap_items: List[str] = lmap(str, range_items)
    itr_items: Iterator[str] = iterator_from(str, range_items)

    itr_items_copy: Iterator[str] = iter(itr_items)

    def print_iterable(iterable: Iterable[str]):
        out_s ="Items: "
        for item in iterable:
            out_s += item
        print(out_s)

    print_iterable(map_items)
    print_iterable(map_items)

    print_iterable(lmap_items)
    print_iterable(lmap_items)

    print_iterable(itr_items)
    print_iterable(itr_items)

    print_iterable(itr_items_copy)
    print_iterable(itr_items_copy)

if __name__ == "__main__":
    main()