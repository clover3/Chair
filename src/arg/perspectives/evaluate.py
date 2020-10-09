

# predictions : List[(Claim, Perspective)] in Ids
# Gold : List[(Claim, List[set(perspectives)]] in Ids
#    get_claim_perspective_dict()
from typing import Tuple, List

from arg.perspectives.load import get_claim_perspective_id_dict, get_perspective_dict
from list_lib import flatten
from misc_lib import average, get_f1, SuccessCounter

perspective = None
claims_d = None


def perspective_getter(pid):
    global perspective
    if perspective is None:
        perspective = get_perspective_dict()
    return perspective[pid]


def get_prec_recll(predicted_perspectives, gold_pids, debug):
    ## In this metrics, it is possible to get precision > 1, as some clusters shares same perspective
    tp = 0
    # if debug:
    #     print(gold_pids)
    # for cluster in gold_pids:
    #     print("-")
    #     for pid in cluster:
    #         print(pid, perspective_getter(pid))
    for prediction in predicted_perspectives:
        pid = prediction['pid']
        valid = False
        for cluster in gold_pids:
            if pid in cluster:
                tp += 1
                valid = True
                break
        if not valid:
            correct_str = "N"
        else:
            correct_str = "Y"
        if debug:
            print(correct_str, prediction['score'], prediction['rationale'], pid, prediction['perspective_text'])
    # r_tp = 0
    # for cluster in gold_pids:
    #     for pid in p_Id_list:
    #         if pid in cluster:
    #             r_tp += 1
    #             break
    prec = tp / len(predicted_perspectives) if len(predicted_perspectives) > 0 else 1
    # I believe correcct : recall = r_tp / len(gold_pids) if len(gold_pids) > 0 else 1
    recall = tp / len(gold_pids) if len(gold_pids) > 0 else 1

    return prec, recall


def get_modified_recall(predicted_perspectives, gold_pids, debug):
    ## In this metrics, it is possible to get precision > 1, as some clusters shares same perspective
    tp = 0
    for prediction in predicted_perspectives:
        pid = prediction['pid']
        valid = False
        for cluster in gold_pids:
            if pid in cluster:
                tp += 1
                valid = True
                break
        if not valid:
            correct_str = "N"
        else:
            correct_str = "Y"
        if debug and False:
            print(correct_str, prediction['score'], prediction['rationale'], pid, prediction['perspective_text'])
    p_Id_list = list([p['pid'] for p in predicted_perspectives])
    r_tp = 0
    for cluster in gold_pids:
        for pid in p_Id_list:
            if pid in cluster:
                r_tp += 1
                break
    if r_tp == 0:
        for cluster in gold_pids:
            print("-")
            for pid in cluster:
                print(pid, perspective_getter(pid))

    prec = tp / len(predicted_perspectives) if len(predicted_perspectives) > 0 else 1
    recall = r_tp / len(gold_pids) if len(gold_pids) > 0 else 1
    return prec, recall


def get_ap(predicted_perspectives, gold_pids, debug):
    ## In this metrics, it is possible to get precision > 1, as some clusters shares same perspective
    # if debug:
    #     print(gold_pids)
    # for cluster in gold_pids:
    #     print("-")
    #     for pid in cluster:
    #         print(pid, perspective_getter(pid))
    def is_correct(pid):
        for cluster in gold_pids:
            if pid in cluster:
                return True
        return False

    tp = 0
    precision_list = []
    for idx, prediction in enumerate(predicted_perspectives):
        pid = prediction['pid']
        if is_correct(pid):
            tp += 1
            n_pred = idx + 1
            prec = tp / n_pred
            precision_list.append(prec)
            correct_str = "Y"
        else:
            correct_str = "N"

        if debug:
            print(correct_str, prediction['score'], prediction['rationale'], pid, prediction['perspective_text'])
    assert tp == len(precision_list)
    ap = average(precision_list) if tp > 0 else 1
    return ap


def get_correctness(predicted_perspectives: List[Tuple[int, int, int]],
                    gold_pids) -> List[int]:
    def get_gold_labels(pid):
        for cluster in gold_pids:
            if pid in cluster:
                return 1
        return 0

    def is_correct(pid, decision):
        gold = get_gold_labels(pid)
        return int(gold == decision)

    correctness_list = list([is_correct(pid, decision) for cid, pid, decision in predicted_perspectives])
    return correctness_list


def get_correctness_list(predictions, debug) -> List[List[int]]:
    gold = get_claim_perspective_id_dict()
    all_correctness_list = []
    for c_Id, prediction_list in predictions:
        gold_pids = gold[c_Id]
        correctness_list: List[int] = get_correctness(prediction_list, gold_pids)
        all_correctness_list.append(correctness_list)
    return all_correctness_list


def get_acc(predictions, debug) -> float:
    all_correctness_list = get_correctness_list(predictions, debug)
    return average(flatten(all_correctness_list))



def evaluate2(predictions):
    gold = get_claim_perspective_id_dict()
    tot_p = tot_r = tot_count = 0
    for c_Id, p_Id_list in predictions:
        gold_pids = gold[c_Id]

        covered = [False for _c in gold_pids]
        for pid in p_Id_list:
            for idx, cluster in enumerate(gold_pids):
                if pid in cluster:
                    covered[idx] = True
        tot_gold = len(covered)
        tot_pred = len(p_Id_list)
        hit = [h for h in covered if h]

        if tot_pred == 0:
            tot_p += 1
        else:
            tot_p += len(hit) / tot_pred

        if tot_gold == 0:
            tot_r += 1
        else:
            tot_r += len(hit) / tot_gold

    mean_p = tot_p / len(predictions)
    mean_r = tot_r / len(predictions)
    mean_f1 = 2 * mean_p * mean_r / (mean_p + mean_r)

    return {
        'precision':mean_p,
        'recall':mean_r,
        'f1': mean_f1
    }


def evaluate(predictions, debug=True):
    gold = get_claim_perspective_id_dict()
    prec_list = []
    recall_list = []
    for c_Id, prediction_list in predictions:
        gold_pids = gold[c_Id]
        claim_text = prediction_list[0]['claim_text']
        if debug:
            print("Claim {}: ".format(c_Id), claim_text)
        prec, recall = get_prec_recll(prediction_list, gold_pids, debug)
        prec_list.append(prec)
        recall_list.append(recall)

    avg_prec = average(prec_list)
    avg_recall = average(recall_list)

    return {
        'precision': avg_prec,
        'recall' : avg_recall,
        'f1': get_f1(avg_prec, avg_recall)
    }


def evaluate_recall(predictions, debug=True):
    gold = get_claim_perspective_id_dict()
    prec_list = []
    recall_list = []
    for c_Id, prediction_list in predictions:
        gold_pids = gold[c_Id]
        claim_text = prediction_list[0]['claim_text']
        if debug:
            print("Claim {}: ".format(c_Id), claim_text)
        prec, recall = get_modified_recall(prediction_list, gold_pids, debug)
        prec_list.append(prec)
        recall_list.append(recall)

    l = sum([1 for r in recall_list if r < 0.001])
    print("zero recall : ", l)
    avg_prec = average(prec_list)
    avg_recall = average(recall_list)

    return {
        'precision': avg_prec,
        'recall' : avg_recall,
        'f1': get_f1(avg_prec, avg_recall)
    }


def evaluate_map(predictions, debug=True):
    ap_list = get_average_precision_list(predictions, debug)
    map = average(ap_list)
    return {'map': map}


def get_average_precision_list(predictions, debug):
    gold = get_claim_perspective_id_dict()
    ap_list = []
    for c_Id, prediction_list in predictions:
        gold_pids = gold[c_Id]
        claim_text = prediction_list[0]['claim_text']
        if debug:
            print("Claim {}: ".format(c_Id), claim_text)
        ap = get_ap(prediction_list, gold_pids, debug)
        ap_list.append(ap)
    return ap_list


def inspect(predictions):
    gold = get_claim_perspective_id_dict()

    suc_counter = SuccessCounter()
    for c_Id, prediction_list in predictions:
        gold_pids = gold[c_Id]

        def is_valid(pid):
            for cluster in gold_pids:
                if pid in cluster:
                    return True
            return False

        top_pred = prediction_list[0]

        if is_valid(top_pred['pid']):
            suc_counter.suc()
        else:
            suc_counter.fail()
            prediction = prediction_list[0]
            claim_text = prediction['claim_text']
            print("Claim {}: ".format(c_Id), claim_text)
            print("{0:.2f} {1} {2}".format(prediction['score'], prediction['rationale'], prediction['perspective_text']))
            print()

    print("P@1", suc_counter.get_suc_prob())