import os
import sys

from arg.perspectives.eval_caches import eval_map
from arg.perspectives.ppnc.ppnc_eval import summarize_score
from cpath import output_path
from misc_lib import exist_or_mkdir


def filter_score_d(score_d):
    valid_list = [499, 629, 894, 311, 628, 704, 23, 286, 469, 666, 591, 731, 191, 664, 952, 842, 598, 707, 457, 166, 192, 17, 509,
     942, 948, 500, 864, 493, 89, 807, 726, 947, 872, 609, 343, 497, 49, 515, 508, 641, 544, 116, 496, 608, 691, 753,
     854, 24, 462, 694, 684, 721, 722, 572, 637, 676, 160, 575, 514, 47, 86, 960, 736, 542, 142, 927, 9, 463, 374, 705,
     910, 776, 511, 730, 745, 838, 921, 638, 180, 112, 525, 817, 579, 34, 835, 194, 837, 464, 735, 831, 453, 489, 501,
     844, 510, 482, 159, 1000, 595, 830, 808, 570, 812, 900, 25, 893, 978, 774, 518, 353, 569, 1004]
    return {k: v for k, v in score_d.items() if k[0] in valid_list}

def main():
    save_name = sys.argv[1]
    out_dir = os.path.join(output_path, "cppnc")
    exist_or_mkdir(out_dir)
    info_file_path = os.path.join(out_dir, "cppnc_triple_all_dev_info")
    pred_file_path = os.path.join(out_dir, save_name + ".score")
    score_d = summarize_score(info_file_path, pred_file_path)
    score_d = filter_score_d(score_d)
    map_score = eval_map("dev", score_d, False)
    print(map_score)


if __name__ == "__main__":
    main()

