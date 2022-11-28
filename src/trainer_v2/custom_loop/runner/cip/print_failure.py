from collections import Counter
from typing import Iterator

import numpy as np

from data_generator.special_tokens import MASK_ID
from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text
from trainer_v2.custom_loop.per_task.cip.cip_common import Comparison
from trainer_v2.custom_loop.runner.cip.cip_stats_from_file import iter_cip_preds


def print_failure(iterator: Iterator[Comparison]):
    tokenizer = get_tokenizer()
    def is_target_failure(ts_g_pred, baseline_pred, label):
        return ts_g_pred != label and baseline_pred == label

    confusion = Counter()
    n_print = 0
    for i, item in enumerate(iterator):
        if i < 54:
            continue
        printed = False
        for ts_probs, ts_info in zip(item.ts_pred_probs, item.ts_input_info_list):
            l_probs, g_probs = ts_probs
            assert len(l_probs[0]) == 3
            assert len(g_probs) == 3
            l_pred1 = np.argmax(l_probs[0])
            l_pred2 = np.argmax(l_probs[1])

            label = item.label
            g_pred = np.argmax(g_probs)
            baseline_pred = np.argmax(item.full_pred_probs)
            if is_target_failure(g_pred, baseline_pred, label):
            #     confusion["{}->{}".format(g_pred, baseline_pred)] += 1
            #     print("Maybe double negation at ", i)
                print()
                if not printed:
                    print("Data ID ", i)
                    print("Prem:", ids_to_text(tokenizer, item.prem))
                    print("Hypo:", ids_to_text(tokenizer, item.hypo))
                    printed = True
                st, ed = ts_info
                h1 = item.hypo[:st] + [MASK_ID] + item.hypo[ed:]
                h2 = item.hypo[st:ed]
                print("{} != {}  , ts != base ".format(g_pred, baseline_pred))
                print("{} H1: {}".format(l_pred1, ids_to_text(tokenizer, h1)))
                print("{} H2: {}".format(l_pred2, ids_to_text(tokenizer, h2)))
                n_print += 1

            if n_print % 100 == 99:
                dummy = input("Press enter")


def main():
    print_failure(iter_cip_preds())
    pass


if __name__ == "__main__":
    main()