from cpath import output_path
from misc_lib import path_join
from trainer_v2.per_project.transparency.misc_common import load_tsv, save_tsv


def main():
    cand2_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "candidate2.tsv")

    cand2_1_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "candidate2_1.tsv")

    cand2 = set(map(tuple, load_tsv(cand2_path)))
    cand2_1 = load_tsv(cand2_1_path)

    cand2_1_new = []
    for pair in cand2_1:
        if tuple(pair) not in cand2:
            cand2_1_new.append(pair)

    print(f"selected {len(cand2_1_new)} from {len(cand2_1)}")

    cand2_1_new_path = path_join(
        output_path, "msmarco", "passage", "align_candidates", "candidate2_1_new.tsv")

    save_tsv(cand2_1_new, cand2_1_new_path)



if __name__ == "__main__":
    main()