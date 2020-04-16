

# predictions : List[(Claim, Perspective)] in Ids
# Gold : List[(Claim, List[set(perspectives)]] in Ids
#    get_claim_perspective_dict()

from arg.perspectives.load import get_claim_perspective_id_dict, get_perspective_dict
from misc_lib import average, get_f1

perspective = None
claims_d = None


def perspective_getter(pid):
    global perspective
    if perspective is None:
        perspective = get_perspective_dict()
    return perspective[pid]


def get_prec_recll(predicted_perspectives, gold_pids):
    ## In this metrics, it is possible to get precision > 1, as some clusters shares same perspective
    tp = 0
    print(gold_pids)
    for cluster in gold_pids:
        print("-")
        for pid in cluster:
            print(pid, perspective_getter(pid))
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


def evaluate(predictions):
    gold = get_claim_perspective_id_dict()
    prec_list = []
    recall_list = []
    for c_Id, prediction_list in predictions:
        gold_pids = gold[c_Id]
        claim_text = prediction_list[0]['claim_text']
        print("Claim: ", claim_text)
        prec, recall = get_prec_recll(prediction_list, gold_pids)
        prec_list.append(prec)
        recall_list.append(recall)

    avg_prec = average(prec_list)
    avg_recall = average(recall_list)

    return {
        'precision':avg_prec,
        'recall':avg_recall,
        'f1': get_f1(avg_prec, avg_recall)
    }



def my():
    pass
