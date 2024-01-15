import heapq
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

class SortedItem:
    def __init__(self, value):
        self.value = value

    def get_key(self):
        return self.value


def merge_sorted_generators(generators: List[Iterator]) -> Iterator:
    """
    Merges multiple sorted generators into a single sorted iterator.

    Args:
    *generators: A variable number of generators, each yielding SortedItem objects.

    Yields:
    SortedItem: The next SortedItem object in sorted order based on its key.
    """
    pq = []
    for idx, gen in enumerate(generators):
        try:
            first_item = next(gen)
            heapq.heappush(pq, (first_item.get_key(), idx, first_item))
        except StopIteration:
            continue

    while pq:
        _, gen_idx, item = heapq.heappop(pq)
        yield item
        try:
            next_item = next(generators[gen_idx])
            heapq.heappush(pq, (next_item.get_key(), gen_idx, next_item))
        except StopIteration:
            continue
#
# # Example usage:
# gen1 = (SortedItem(i) for i in [1, 3, 5])
# gen2 = (SortedItem(i) for i in [2, 4, 6])
# merged_gen = merge_sorted_generators(gen1, gen2)
# for item in merged_gen:
#     print(item.get_key())
#
# # Note: Uncomment the above lines to test the function with example generators.
