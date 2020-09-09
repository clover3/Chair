import os
import sys

from arg.perspectives.eval_caches import get_ap_list_from_score_d
from arg.perspectives.ppnc.ppnc_eval import summarize_score
from cpath import output_path
from misc_lib import exist_or_mkdir
from tab_print import print_table


def main():
    save_name = sys.argv[1]
    out_dir = os.path.join(output_path, "cppnc")
    exist_or_mkdir(out_dir)
    info_file_path = os.path.join(out_dir, save_name + ".info")
    pred_file_path = os.path.join(out_dir, save_name + ".score")
    score_d = summarize_score(info_file_path, pred_file_path)
    # load pre-computed perspectives
    split = "dev"
    ap_list, cids = get_ap_list_from_score_d(score_d, split)
    print_table(zip(cids, ap_list))


if __name__ == "__main__":
    main()
