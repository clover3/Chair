import os

from cache import load_list_from_jsonl
from cpath import output_path
from data_generator.tokenizer_wo_tf import ids_to_text, get_tokenizer
from misc_lib import pause_hook
from trainer_v2.per_project.cip.cip_common import split_into_two
from trainer_v2.per_project.cip.precomputed_cip import get_cip_pred_splits_iter
import numpy as np
from typing import List, Iterable, Callable, Dict, Tuple, Set




def main():
    tokenizer = get_tokenizer()
    split, itr, _ = get_cip_pred_splits_iter()[0]
    save_path = os.path.join(output_path, "nlits", "nli_cip2_0_train_val_scores")
    items = load_list_from_jsonl(save_path, lambda x: x)
    for comparison, scores in pause_hook(zip(itr, items), 20):
        fail_probs = np.array(scores)[:, 1]
        rank = np.argsort(fail_probs)
        hypo: List[int] = comparison.hypo
        h = ids_to_text(tokenizer, hypo)
        print()
        print("H:", h)
        for i in rank:
            st, ed = comparison.ts_input_info_list[i]
            hypo1, hypo2 = split_into_two(hypo, st, ed)
            s = fail_probs[i]
            h1 = ids_to_text(tokenizer, hypo1)
            h2 = ids_to_text(tokenizer, hypo2)
            print(f"{s:.2f} {h1} \t {h2}")


if __name__ == "__main__":
    main()