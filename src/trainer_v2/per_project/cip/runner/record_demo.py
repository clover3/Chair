from collections import OrderedDict
from typing import List, Callable, Iterator

import numpy as np

from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text
from misc_lib import pause_hook
from trainer_v2.per_project.cip.cip_common import split_into_two
from trainer_v2.per_project.cip.precomputed_cip import get_cip_pred_splits_iter
from trainer_v2.per_project.cip.tfrecord_gen import LabeledInstance, SelectOneToOne


def iter_labeld_instance_by_one_one():
    todo = get_cip_pred_splits_iter()
    item_selector = SelectOneToOne()
    for split, itr, src_size in todo:
        labeled_instance_itr: Iterator[LabeledInstance] = item_selector.select(itr)
        yield from labeled_instance_itr


def main():
    tokenizer = get_tokenizer()
    last_h = ""

    def show_for_item(e: LabeledInstance):
        hypo: List[int] = e.comparison.hypo
        hypo1, hypo2 = split_into_two(hypo, e.st, e.ed)
        h = ids_to_text(tokenizer, hypo)
        h1 = ids_to_text(tokenizer, hypo1)
        h2 = ids_to_text(tokenizer, hypo2)

        found_i = -1
        for i in range(len(e.comparison.ts_pred_probs)):
            st, ed = e.comparison.ts_input_info_list[i]
            if e.st == st and e.ed == ed:
                found_i = i
                break
        ts_probs = e.comparison.ts_pred_probs[found_i]
        local_d, global_d = ts_probs
        def prob_to_str(prob):
            pred = np.argmax(prob)
            return ['E', "N", "C"][pred]
            # return ", ".join(map(two_digit_float, prob))
        nonlocal last_h
        if h != last_h:
            p = ids_to_text(tokenizer, e.comparison.prem)
            print("")
            print("P:", p)
            print("H:", h)
            print("BASE:", prob_to_str(e.comparison.full_pred_probs))
            last_h = h

        s2 = "{} {} {}".format(prob_to_str(global_d), prob_to_str(local_d[0]), prob_to_str(local_d[1]))
        print(f"{e.label} {s2} {h1} \t {h2}")

    itr = pause_hook(iter_labeld_instance_by_one_one(), 100)
    for e in itr:
        show_for_item(e)


if __name__ == "__main__":
    main()
