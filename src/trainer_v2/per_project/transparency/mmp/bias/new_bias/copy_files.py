from cpath import output_path
from misc_lib import path_join
import shutil

from trainer_v2.per_project.transparency.mmp.bias.new_bias.path_helper import get_rel_table_score_path


def main():
    score_dir = path_join(output_path, "mmp", "mct6_pep_tt17_10000", )
    log_save_path = path_join(output_path, "mmp", "bias", "brand", "product_category_selected.txt")

    terms: list[tuple[str, int]] = [line.strip().split() for line in open(log_save_path, "r")]
    for idx_s, term in terms:
        loc = int(idx_s) - 1
        score_path = path_join(score_dir, "%d.txt" % loc)
        target_path = get_rel_table_score_path(term)
        shutil.copy(score_path, target_path)


if __name__ == "__main__":
    main()
