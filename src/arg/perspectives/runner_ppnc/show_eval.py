import os

from arg.perspectives.ppnc.ppnc_eval import summarize_score, eval_map
from cpath import output_path


def main():
    info_dir = "/mnt/nfs/work3/youngwookim/job_man/ppnc_50_pers_val"
    #prediction_file = os.path.join(output_path, "ppnc_50_val_prediction")
    prediction_file = os.path.join(output_path, "ppnc_val")
    score_d = summarize_score(info_dir, prediction_file)
    map_score = eval_map("train", score_d)
    print(map_score)


if __name__ == "__main__":
    main()

