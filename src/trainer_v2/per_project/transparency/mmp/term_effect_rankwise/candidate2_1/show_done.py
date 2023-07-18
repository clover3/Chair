import os

from cpath import output_path
from misc_lib import path_join, TimeEstimator
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand2_1_path_helper
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_fidelity_save_path2


def compress_sequence(input_list):
    output_list = []
    start = input_list[0]
    end = input_list[0]

    for i in range(1, len(input_list)):
        if input_list[i] == end + 1:
            end = input_list[i]
        else:
            if start == end:
                output_list.append(str(start))
            else:
                output_list.append(f"{start}-{end}")
            start = end = input_list[i]

    if start == end:
        output_list.append(str(start))
    else:
        output_list.append(f"{start}-{end}")

    return output_list


def main():
    ph = get_cand2_1_path_helper()

    cand_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "candidate2_1.tsv")
    todo_list = [line.strip() for line in open(cand_path, "r")]
    done = []
    for i in range(len(todo_list)):
        q_term, d_term = todo_list[i].split()
        save_path = ph.get_fidelity_save_path(q_term, d_term)
        if os.path.exists(save_path):
            done.append(i)

    print(", ".join(compress_sequence(done)))


if __name__ == "__main__":
    main()
