import os
import sys

from arg.perspectives.eval_caches import eval_map
from arg.perspectives.ppnc.ppnc_eval import summarize_score
from cpath import output_path
from misc_lib import exist_or_mkdir


def main():
    save_name = sys.argv[1]
    out_dir = os.path.join(output_path, "cppnc")
    exist_or_mkdir(out_dir)
    info_file_path = os.path.join(out_dir, sys.argv[2])
    pred_file_path = os.path.join(out_dir, save_name + ".score")

    debug = False
    split = "dev"
    if len(sys.argv) > 3:
        if sys.argv[3] == "debug":
            debug = True
    if len(sys.argv) > 4:
        if sys.argv[4] == "train":
            split = "train"
    score_d = summarize_score(info_file_path, pred_file_path)
    map_score = eval_map(split, score_d, debug)
    print(map_score)


def save_to_trec_format():
    pass


if __name__ == "__main__":
    main()

