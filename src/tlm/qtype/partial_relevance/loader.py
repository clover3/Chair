import os
from typing import List

from cache import load_pickle_from
from cpath import output_path
from tlm.qtype.partial_relevance.eval_data_structure import RelatedEvalInstance


def load_dev_small_problems() -> List[RelatedEvalInstance]:
    save_path = os.path.join(output_path, "qtype", "MMDE_dev_problems_sm.pickle")
    items: List[RelatedEvalInstance] = load_pickle_from(save_path)
    return items


def load_dev_problems() -> List[RelatedEvalInstance]:
    save_path = os.path.join(output_path, "qtype", "MMDE_dev_problems.pickle")
    items: List[RelatedEvalInstance] = load_pickle_from(save_path)
    return items


def main():
    return NotImplemented


if __name__ == "__main__":
    main()