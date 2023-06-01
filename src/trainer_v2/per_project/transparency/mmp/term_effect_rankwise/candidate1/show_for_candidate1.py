import sys
from cpath import output_path
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.runner.summarize_te import show_te


def main():
    save_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "candidate1.tsv")

    todo_list = [line.strip() for line in open(save_path, "r")]
    for i in range(100):
        q_term, d_term = todo_list[i].split()
        print(q_term, d_term)
        show_te(q_term, d_term)



if __name__ == "__main__":
    main()