import os
import sys
from collections import defaultdict
from typing import Dict, Tuple, List

import scipy.special

from arg.perspectives.ppnc.collect_score import load_combine_info_jsons
from arg.perspectives.types import CPIDPair
from cpath import output_path
from misc_lib import exist_or_mkdir
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer
from visualize.html_visual import HtmlVisualizer


def collect_info(prediction_file, info: Dict, logit_to_score) -> Dict[CPIDPair, List[Tuple[float, float, Dict]]]:
    data = EstimatorPredictionViewer(prediction_file)
    print("Num data ", data.data_len)
    out_d: Dict[CPIDPair, List[Tuple[float, float, Dict]]] = defaultdict(list)


    for entry in data:
        logits = entry.get_vector("logits")
        score = logit_to_score(logits)
        rel_score = entry.get_vector("rel_score")[0]
        data_id = entry.get_vector("data_id")[0]
        try:
            cur_info = info[str(data_id)]
            cid = cur_info['cid']
            pid = cur_info['pid']
            cpid = CPIDPair((cid, pid))
            out_d[cpid].append((score, rel_score, cur_info))
        except KeyError as e:
            print("Key error")
            print("data_id", data_id)
            pass
    return out_d



def show(all_info):
    html = HtmlVisualizer("cppnc.html")
    cnt = 0
    for cpid, value in all_info.items():
        score, rel_score, info = value[0]
        html.write_headline("Claim {}: {}".format(info['cid'], info['c_text']))
        html.write_headline("Perspective: " + info['p_text'])

        for score, rel_score, info in value:
            html.write_headline("score: {}".format(score))
            html.write_headline("rel_score: {}".format(rel_score))
            html.write_paragraph(" ".join(info['passage']))
        cnt += 1

        if cnt > 10000:
            break





def logit_to_score_softmax(logit):
    return scipy.special.softmax(logit)[1]


def main():
    save_name = sys.argv[1]
    out_dir = os.path.join(output_path, "cppnc")
    exist_or_mkdir(out_dir)
    info_file_path = os.path.join(out_dir, save_name + ".info")
    pred_file_path = os.path.join(out_dir, save_name + ".score")
    info = load_combine_info_jsons(info_file_path)

    all_info = collect_info(pred_file_path, info, logit_to_score_softmax)
    show(all_info)

    pass


if __name__ == "__main__":
    main()