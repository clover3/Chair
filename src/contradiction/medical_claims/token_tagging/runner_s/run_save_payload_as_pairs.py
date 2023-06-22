from contradiction.medical_claims.token_tagging.problem_loader import AlamriProblem, load_alamri_problem
from typing import List, Iterable, Callable, Dict, Tuple, Set
from cpath import output_path
from misc_lib import path_join


def save_as_text():
    problems: List[AlamriProblem] = load_alamri_problem()

    save_path = path_join(output_path, "alamri_annotation1", "plain_payload.txt")
    f = open(save_path, "w")
    for p in problems:
        f.write("{}\t{}\n".format(p.text1, p.text2))


if __name__ == "__main__":
    save_as_text()
