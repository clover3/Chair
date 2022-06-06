from collections import Counter
from typing import List, Callable, Tuple
from typing import NamedTuple

from bert_api import SegmentedText
from bert_api.segmented_instance.segmented_text import seg_to_text
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import pick1
from trainer_v2.chair_logging import c_log


class Subsequence(NamedTuple):
    segmented_text: SegmentedText
    parent: SegmentedText
    parent_drop_indices: List[int]
    # segmented_text is acquired by dropping parent_drop_indices from parent

    def get_seg_len(self):
        return self.segmented_text.get_seg_len()


def do_local_search_addition(init: Subsequence, get_score: Callable[[SegmentedText], float]) -> Subsequence:
    tokenizer = get_tokenizer()
    stop = len(init.parent_drop_indices) == 0
    best_score: float = get_score(init.segmented_text)
    best_sub: Subsequence = init
    n_try_wo_improvement = 0
    while not stop:
        new_point: Subsequence = add_one_more(best_sub)
        print("new_point.parent_drop_indices:  {}".format(len(new_point.parent_drop_indices)))
        print("-> ", seg_to_text(tokenizer, new_point.segmented_text))
        new_score = get_score(new_point.segmented_text)

        if new_score > best_score:
            best_score = new_score
            best_sub: Subsequence = new_point
            n_try_wo_improvement = 0
        else:
            n_try_wo_improvement += 1

        if n_try_wo_improvement > 2 * len(best_sub.parent_drop_indices):
            stop = True
        if len(best_sub.parent_drop_indices) == 0:
            stop = True

    return best_sub


def add_one_more(subsequence: Subsequence) -> Subsequence:
    add_index = pick1(subsequence.parent_drop_indices)
    new_drop_indices = [i for i in subsequence.parent_drop_indices if i != add_index]
    new_segmented_text = subsequence.parent.get_dropped_text(new_drop_indices)
    new_subsequence = Subsequence(new_segmented_text, subsequence.parent, new_drop_indices)
    return new_subsequence


def do_local_search(init: Subsequence,
                    get_score: Callable[[Subsequence], float],
                    modify_fn: Callable[[Subsequence], Subsequence],
                    left_better: Callable[[float, float], bool],
                    terminate_condition_fn: Callable[[int, Subsequence, float], bool],
                    ) -> Tuple[Subsequence, Counter]:
    stop = False
    best_score: float = get_score(init)
    best_sub: Subsequence = init
    n_try_wo_improvement = 0
    info = Counter()
    while not stop:
        new_point: Subsequence = modify_fn(best_sub)
        new_score = get_score(new_point)
        info['n_call'] += 1

        if left_better(new_score, best_score):
            info['n_update'] += 1
            best_score = new_score
            c_log.debug(f"After {n_try_wo_improvement + 1} trials, Update ({best_sub.get_seg_len()}, {best_score:.2f}) "
                        f"-> ({new_point.get_seg_len()}, {new_score:.2f})")
            best_sub: Subsequence = new_point
            n_try_wo_improvement = 0
        else:
            n_try_wo_improvement += 1

        if terminate_condition_fn(n_try_wo_improvement, best_sub, new_score):
            stop = True

    return best_sub, info
