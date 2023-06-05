import os

from cpath import output_path
from misc_lib import path_join, TimeEstimator
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_fidelity_save_path2


def main():
    cand_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "candidate2.tsv")
    todo_list = [line.strip() for line in open(cand_path, "r")]
    missing = []
    missing_100 = set()
    for i in range(0, 10000):
        q_term, d_term = todo_list[i].split()
        save_path = get_fidelity_save_path2(q_term, d_term)
        if not os.path.exists(save_path):
            missing.append(i)
            missing_100.add(i - i % 100)

    print(f"{len(missing)} missing")
    missing_100 = list(missing_100)
    missing_100.sort()
    print(missing_100)


if __name__ == "__main__":
    main()
