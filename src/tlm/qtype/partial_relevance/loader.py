import os
from collections import Counter
from typing import List

from cache import load_list_from_jsonl
from cpath import output_path
from misc_lib import average
from contradiction.alignment.data_structure.related_eval_instance import RelatedEvalInstance


def load_dev_small_problems() -> List[RelatedEvalInstance]:
    save_path = os.path.join(output_path, "qtype", "MMDE_dev_sm_problems.json")
    items: List[RelatedEvalInstance] = load_list_from_jsonl(save_path, RelatedEvalInstance.from_json)
    return items


def load_dev_problems() -> List[RelatedEvalInstance]:
    save_path = os.path.join(output_path, "qtype", "MMDE_dev_problems.json")
    items: List[RelatedEvalInstance] = load_list_from_jsonl(save_path, RelatedEvalInstance.from_json)
    return items


def load_mmde_problem(dataset_name) -> List[RelatedEvalInstance]:
    save_path = os.path.join(output_path, "qtype", "MMDE_{}_problems.json".format(dataset_name))
    items: List[RelatedEvalInstance] = load_list_from_jsonl(save_path, RelatedEvalInstance.from_json)
    return items


def main():
    counter = Counter()
    for p in load_dev_small_problems():
        qid, _ = p.problem_id.split("-")
        counter[qid] += 1

    print("{} queries, {} per query".format(len(counter), average(counter.values())))


if __name__ == "__main__":
    main()