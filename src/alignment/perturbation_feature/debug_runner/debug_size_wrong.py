from typing import List

from alignment import RelatedEvalInstance
from alignment.nli_align_path_helper import load_mnli_rei_problem


def main():
    problem_id = "60529c"
    problem_list: List[RelatedEvalInstance] = load_mnli_rei_problem(dataset_name)
    return NotImplemented


if __name__ == "__main__":
    main()