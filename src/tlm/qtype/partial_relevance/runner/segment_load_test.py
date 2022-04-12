import os
from typing import List

from cache import load_pickle_from
from cpath import output_path
from alignment.data_structure.related_eval_instance import RelatedEvalInstance


def main():
    save_path = os.path.join(output_path, "qtype", "MMDE_dev_problems.pickle")
    items: List[RelatedEvalInstance] = load_pickle_from(save_path)
    for e in items:
        print(e.problem_id, e.seg_instance.text2_seg_indices)


if __name__ == "__main__":
    main()
