from collections import OrderedDict
from typing import List, Callable

import numpy as np

from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text
from trainer_v2.custom_loop.per_task.cip.cip_common import split_into_two
from trainer_v2.custom_loop.per_task.cip.tfrecord_gen import LabeledInstance, build_encoded, \
    SelectOneToOne, ItemSelector, encode_three


def seq300wrap(
        name: str,
        selector: ItemSelector,
        encode_fn_inner: Callable[[int, LabeledInstance], OrderedDict]):
    seq_length = 300
    print("Dataset name={}".format(name))
    tokenizer = get_tokenizer()
    last_h = ""
    def encode_fn(e: LabeledInstance) -> OrderedDict:
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
        ts_global_pred = np.argmax(global_d)
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
        return encode_fn_inner(seq_length, e)

    build_encoded(name, selector, encode_fn)


def main():
    seq300wrap("cip_dummy", SelectOneToOne(), encode_three)

#

if __name__ == "__main__":
    main()
