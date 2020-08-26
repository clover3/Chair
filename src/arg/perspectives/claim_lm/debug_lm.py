from typing import Set

from arg.perspectives.evaluate import get_ap
from arg.perspectives.load import get_claim_perspective_id_dict
from list_lib import flatten, lmap, foreach
from misc_lib import average


def debug_failture(predictions):
    gold = get_claim_perspective_id_dict()
    ap_list = []
    for c_Id, prediction_list in predictions:
        gold_pids = gold[c_Id]
        gold_pids_set : Set[int] = set(flatten(gold_pids))
        claim_text = prediction_list[0]['claim_text']
        print("Claim {}: ".format(c_Id), claim_text)
        correctness_list = lmap(lambda p: p['pid'] in gold_pids_set, prediction_list)
        ap = get_ap(prediction_list, gold_pids, False)

        if not any(correctness_list): # all wrong
            continue

        if ap > 0.9:
            continue

        def print_line(prediction):
            pid = prediction['pid']
            correct = pid in gold_pids_set
            if correct:
                correct_str = "Y"
            else:
                correct_str = "N"

            score = prediction['score']
            print(correct_str, score, score.name, prediction['perspective_text'])

        foreach(print_line, prediction_list)
        ap_list.append(ap)

    map = average(ap_list)
    return {'map': map}
