import os.path

from misc_lib import TEL, path_join
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.candidate2_1.show_done import compress_sequence
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_fidelity_save_name
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_cand2_1_path_helper, \
    get_cand2_1_spearman_path_helper


def print_missing_fidelity(term_pair_list, fidelity_save_dir):
    n = len(term_pair_list)
    missing_item = []
    for i in range(n):
        q_term, d_term = term_pair_list[i]
        try:
            save_name = get_fidelity_save_name(q_term, d_term)
            save_path = path_join(fidelity_save_dir, save_name)
            if not os.path.exists(save_path):
                missing_item.append(i)
        except ValueError:
            pass
        except FileNotFoundError:
            pass

    if not missing_item:
        print("no missing items")
    else:
        print(", ".join(compress_sequence(missing_item)))



def main():
    # ph = get_cand2_1_path_helper()
    ph = get_cand2_1_spearman_path_helper()
    cand_path = ph.per_pair_candidates.candidate_pair_path
    term_pair_list = [line.strip().split() for line in open(cand_path, "r")]
    print_missing_fidelity(
        term_pair_list,
        ph.per_pair_candidates.fidelity_save_dir)


if __name__ == "__main__":
    main()
