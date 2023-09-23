import csv

from misc_lib import TEL, path_join
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_fidelity_save_name


def collect_scores_and_save(term_pair_list, fidelity_save_dir, save_path):
    f_out = csv.writer(open(save_path, "w", encoding="utf-8"), dialect='excel-tab')
    for todo in TEL(term_pair_list):
        q_term, d_term = todo
        try:
            save_name = get_fidelity_save_name(q_term, d_term)
            save_path = path_join(fidelity_save_dir, save_name)
            score = float(open(save_path, "r").read())
            row = [q_term, d_term, score]
            f_out.writerow(row)
        except ValueError:
            pass
        except FileNotFoundError:
            pass


def collect_scores_and_save2(term_pair_list, fidelity_save_dir, save_path):
    f_out = csv.writer(open(save_path, "w", encoding="utf-8"), dialect='excel-tab')
    for idx, todo in enumerate(term_pair_list):
        q_term, d_term = todo
        try:
            save_name = f"{idx}"
            save_path = path_join(fidelity_save_dir, save_name)
            score = float(open(save_path, "r").read())
            row = [q_term, d_term, score]
            f_out.writerow(row)
        except ValueError as e:
            pass
        except FileNotFoundError as e:
            pass


def collect_compare_scores(
        term_pair_list, fidelity_save_dir1, fidelity_save_dir2, save_path):
    f_out = csv.writer(open(save_path, "w", encoding="utf-8"), dialect='excel-tab')
    for todo in TEL(term_pair_list):
        q_term, d_term = todo

        def load_score(fidelity_save_dir):
            try:
                save_name = get_fidelity_save_name(q_term, d_term)
                save_path = path_join(fidelity_save_dir, save_name)
                score = float(open(save_path, "r").read())
                return score
            except ValueError:
                pass
            except FileNotFoundError:
                pass
            return "-"

        score1 = load_score(fidelity_save_dir1)
        score2 = load_score(fidelity_save_dir2)
        row = [q_term, d_term, score1, score2]
        f_out.writerow(row)
