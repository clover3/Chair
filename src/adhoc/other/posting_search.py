from typing import List, Iterable, Callable, Dict, Tuple, Set

def linear_search(arr, low, high, x):
    ed = min(len(arr), high + 1)
    for i in range(low, ed):
        if arr[i] == x:
            return i, i+1
        elif arr[i] > x:
            return -1, i

    return -1, ed


def join_postings_smart(doc_ids: List[int], postings: List[int]) -> List[int]:
    # Return indices of posting that doc_ids appear
    skip = 100
    output = []
    infinite = 1e30

    class State:
        def __init__(self, cur_idx, next_idx):
            self.cur_idx = cur_idx
            self.next_idx = next_idx
            self.cur_val = postings[self.cur_idx]
            if next_idx == len(postings):
                self.next_val = infinite
            else:
                self.next_val = postings[next_idx]

        def get_cur_val(self):
            return self.cur_val

        def get_next_val(self):
            return self.next_val

        def skip(self):
            if self.next_idx == len(postings):
                raise StopIteration

            cur_idx = self.next_idx
            next_idx = min(cur_idx + skip, len(postings))
            return State(cur_idx, next_idx)

        @classmethod
        def from_cur_idx(cls, next_i):
            if next_i == len(postings):
                raise StopIteration
            cur_idx = next_i
            next_idx = min(cur_idx + skip, len(postings))
            return State(cur_idx, next_idx)

    try:
        state = State(0, skip)
        for target_doc_id in doc_ids:
            while state.get_next_val() < target_doc_id:
                # we would skip until skip val is equal or larger than doc_id
                state = state.skip()

            low = state.cur_idx
            high = state.next_idx-1
            if target_doc_id < state.get_cur_val():
                found_idx = -1
                next_i = state.cur_idx
            elif state.get_cur_val() == target_doc_id:
                found_idx = state.cur_idx
                next_i = found_idx + 1
            elif state.get_cur_val() < target_doc_id:
                found_idx, next_i = linear_search(postings, low, high, target_doc_id)
            else:
                assert False

            if found_idx >= 0:
                output.append(found_idx)

            if next_i >= len(postings):
                break
            state = State.from_cur_idx(next_i)
    except StopIteration:
        pass
    return output

