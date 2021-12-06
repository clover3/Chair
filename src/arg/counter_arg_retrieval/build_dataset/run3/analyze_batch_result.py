import pickle
import sys
from collections import Counter
from typing import List, Dict, Tuple

from arg.counter_arg_retrieval.build_dataset.run3.ca2_2_parser import load_file
from arg.counter_arg_retrieval.build_dataset.run3.hit_edit import consistency_check, apply_patch, load_patch_data
from cache import load_pickle_from
from mturk.parse_util import HitResult
from tab_print import save_table_as_csv, tab_print


def apply_fix():
    batch_result_path = "C:\\work\\Code\\Chair\\output\\ca_building\\run3\\batch_result\\Batch_341257_batch_results.csv"
    patch_path = "C:\\work\\Code\\Chair\\output\\ca_building\\run3\\batch_result\\Batch_341257_batch_results.csv.patch.csv"
    output_path = "C:\\work\\Code\\Chair\\output\\ca_building\\run3\\batch_result\\Batch_341257_batch_results.csv.patched.pickle"
    hits = load_file(batch_result_path)
    print("{} hits loaded".format(len(hits)))
    patch_data: Dict[str, List[Tuple[str, str]]] = load_patch_data(patch_path)
    apply_patch(hits, patch_data)
    print("Saved {} hits".format(len(hits)))
    pickle.dump(hits, open(output_path, "wb"))


def generate_fix_candidates():
    batch_result_path = sys.argv[1]
    hits = load_file(batch_result_path)
    out_payload = consistency_check(hits)
    save_path = batch_result_path + ".to_fix.csv"
    save_table_as_csv(out_payload, save_path)


def main2():
    pickle_path = "C:\\work\\Code\\Chair\\output\\ca_building\\run3\\batch_result\\Batch_341257_batch_results.csv.patched.pickle"
    # Show statistics
    hits: List[HitResult] = load_pickle_from(pickle_path)
    # targets = ["Arg_Oppose_P", "Info_Oppose_P"]
    targets = ["Info_Oppose_P"]
    for target in targets:
        cnt = 0
        counter_pos = Counter()
        counter_all = Counter()
        for h in hits:
            perspective = h.inputs['perspective']
            if h.outputs[target]:
                counter_pos[perspective] += 1
                cnt += 1
            counter_all[perspective] += 1
        for key, cnt in counter_all.items():
            tab_print(key, counter_pos[key], cnt)


if __name__ == "__main__":
    main2()
