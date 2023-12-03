import time
from dataclasses import dataclass
from typing import List, Iterable, Callable, Dict, Tuple, Set

from cache import load_pickle_from, load_from_pickle
from list_lib import list_equal


def binary_search(arr, low, high, x):
    num_ops = 0
    while high >= low:
        mid = (high + low) // 2
        num_ops += 1
        # If element is present at the middle itself
        try:
            if arr[mid] == x:
                return mid, low
            # If element is smaller than mid, then it can only
            # be present in left subarray
            elif arr[mid] > x:
                high = mid - 1
            # Else the element can only be present in right subarray
            else:
                low = mid + 1
        except Exception:
            print(high, low, mid)
            raise

    # Element is not present in the array
    return -1, low


def binary_search2(arr, low, high, x):
    num_ops = 0
    found_idx = -1

    while high >= low:
        mid = (high + low) // 2
        num_ops += 1
        # If element is present at the middle itself
        if arr[mid] == x:
            found_idx = mid
            break
        # If element is smaller than mid, then it can only
        # be present in left subarray
        elif arr[mid] > x:
            high = mid - 1
        # Else the element can only be present in right subarray
        else:
            low = mid + 1

    # Element is not present in the array
    return found_idx, low, num_ops


def join_postings(doc_ids: List[int], postings: List[int]) -> List[int]:
    # Return indices of posting that doc_ids appear
    num_ops = 0

    skip_size = 100
    st = 0
    n = len(postings)
    output = []
    for doc_id in doc_ids:
        # print("We want to find", doc_id)
        # print("st", st,)
        ed = st + skip_size

        while ed < n and postings[ed] < doc_id:  # We don't need to consider posting until ed
            num_ops += 1
            # ~ ed will not have doc_id,
            ed += skip_size
            st = ed - skip_size + 1

        # now posting[ed-skip] < doc_id <= posting[ed]
        #   or ed >= n, so
        if postings[ed] == doc_id:
            output.append(ed)
        else:
        # now posting[st] <= doc_id < posting[ed]

            ed = min(ed, n-1)
            # print("after skip", st, ed, postings[st], postings[ed])
            num_ops += 1
            if doc_id < postings[ed]:
                idx, low, num_ops_add = binary_search(postings, st, ed, doc_id)
                num_ops += num_ops_add
                if idx >= 0:
                    output.append(idx)
                    st = idx + 1
                else:
                    if low > st:
                        st = low
    print(f"{num_ops} ops")
    return output


def linear_search(arr, low, high, x):
    num_ops = 0
    ed = min(len(arr), high + 1)
    for i in range(low, ed):
        num_ops += 1
        if arr[i] == x:
            return i, i+1, num_ops
        elif arr[i] > x:
            return -1, i, num_ops

    return -1, ed, num_ops


def join_postings_smart(doc_ids: List[int], postings: List[int], option="linear") -> List[int]:
    # Return indices of posting that doc_ids appear
    skip = 100
    num_ops = 0
    output = []
    infinite = 1e30
    skip_used = 0
    num_ops += 1

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
                num_ops += 1
                skip_used += 1

            low = state.cur_idx
            high = state.next_idx-1
            if target_doc_id < state.get_cur_val():
                found_idx = -1
                next_i = state.cur_idx
            elif state.get_cur_val() == target_doc_id:
                found_idx = state.cur_idx
                next_i = found_idx + 1
            elif state.get_cur_val() < target_doc_id:
                # print(f"target_doc_id={target_doc_id}, state.get_cur_val()={state.get_cur_val()}")
                if option == "binary":
                    found_idx, next_i, num_ops_add = binary_search2(postings, low+1, high, target_doc_id)
                    # print(f"Binary search for {high-low+1} items ({low},{high}) took {num_ops_add} ops, next_i={next_i}")
                elif option == "linear":
                    found_idx, next_i, num_ops_add = linear_search(postings, low, high, target_doc_id)
                    # print(f"Linear search for {high-low+1} items ({low},{high}) took {num_ops_add} ops")
                else:
                    raise ValueError()
                num_ops += num_ops_add
            else:
                assert False

            if found_idx >= 0:
                output.append(found_idx)

            if next_i >= len(postings):
                break
            state = State.from_cur_idx(next_i)
    except StopIteration:
        pass

    print(f"Smart: {num_ops} ops")
    print(f"skip_used: {skip_used}")
    return output


def join_postings_naive(doc_ids: List[int], postings: List[int]) -> List[int]:
    # Return indices of posting that doc_ids appear
    st = 0
    ed = len(postings)

    num_ops = 0
    output = []
    for doc_id in doc_ids:
        for i in range(st, ed):
            num_ops += 1
            if postings[i] == doc_id:
                output.append(i)
                st = i + 1
                break
            elif postings[i] > doc_id:
                st = i
                break
    print(f"Naive: {num_ops} ops")
    return output


def main():
    def convert_to_int_ids(postings):
        return [int(doc_id) for doc_id, value in postings]

    posting1 = convert_to_int_ids(load_from_pickle("posting_question_mark"))
    posting2 = convert_to_int_ids(load_from_pickle("posting_water"))
    posting3 = convert_to_int_ids(load_from_pickle("posting_throb"))
    posting4 = convert_to_int_ids(load_from_pickle("posting_scratchy"))

    max_idx = max(posting1 + posting2 + posting3 + posting4)

    n_query_doc_ids = 1000
    import random
    query_posting = random.sample(range(1, max_idx), n_query_doc_ids)

    for posting in [posting1, posting2, posting4]:
        print("target posting len = {}".format(len(posting)))
        st = time.time()
        out_smart_binary = join_postings_smart(query_posting, posting, "binary")
        t1 = time.time()
        out_smart_linear = join_postings_smart(query_posting, posting, "linear")
        t2 = time.time()
        out_naive = join_postings_naive(query_posting, posting)
        t3 = time.time()
        print("output equal: ", list_equal(out_smart_binary, out_naive))
        print("output equal: ", list_equal(out_smart_binary, out_smart_linear))
        print("out_smart_binary: ", t1 - st)
        print("out_smart_linear: ", t2 - t1)
        print("out_naive: ", t3 - t2)



if __name__ == "__main__":
    main()



