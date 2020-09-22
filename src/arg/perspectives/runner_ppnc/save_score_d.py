import os
import sys

from arg.perspectives.ppnc.ppnc_eval import summarize_score
from cache import save_to_pickle
from cpath import output_path
from misc_lib import exist_or_mkdir


def main():
    save_name = sys.argv[1]
    out_dir = os.path.join(output_path, "cppnc")
    exist_or_mkdir(out_dir)
    info_file_path = os.path.join(out_dir, sys.argv[2])
    pred_file_path = os.path.join(out_dir, save_name + ".score")
    score_d = summarize_score(info_file_path, pred_file_path)
    save_to_pickle(score_d, "score_d")
    print("Saved as 'score_d'")






if __name__ == "__main__":
    main()

