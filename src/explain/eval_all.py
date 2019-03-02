
from evaluation import *
from cache import *
from data_generator.NLI.nli import *

def eval(pred_list, gold_list, only_prem):
    if len(pred_list) != len(gold_list):
        print("Warning")
        print("pred len={}".format(len(pred_list)))
        print("gold len={}".format(len(gold_list)))

    if only_prem:
        p1_p, p1_h = p_at_k_list_ind(pred_list, gold_list, 1)
        #p_at_20 = p_at_k_list(pred_list, gold_list, 20)
        p_auc, h_auc = PR_AUC_ind(pred_list, gold_list)
        MAP_p, MAP_h = MAP_ind(pred_list, gold_list)
        scores = {
            "P@1": p1_p,
            "AUC": p_auc,
            "MAP":MAP_p,
        }
        return scores
    else:
        p1 = p_at_k_list(pred_list, gold_list, 1)
        # p_at_20 = p_at_k_list(pred_list, gold_list, 20)
        auc_score = PR_AUC(pred_list, gold_list)
        MAP_score = MAP(pred_list, gold_list)
        scores = {
            "P@1": p1,
            "AUC": auc_score,
            "MAP": MAP_score,
        }
        return scores

def load_gold(data_id):
    if data_id == 'conflict':
        data= load_mnli_explain_0()
        result = []
        for entry in data:
            result.append((entry['p_explain'],  entry['h_explain']))
    elif data_id == 'match':
        data = load_nli_explain_1(data_id)
        result = []
        for entry in data:
            result.append((entry[2], entry[3]))
    elif data_id.startswith("conflict_"):
        data = load_nli_explain_3("conflict_0_99", "conflict")
        result = []
        for entry in data:
            result.append((entry[2], entry[3]))

    return result


def run_analysis():
    data_id = "conflict_0_99"
    data_id = "conflict"
    data_id = "match"
    prem_only = data_id.startswith("match")
    gold_list = load_gold(data_id)
    def p_at_k(rank_list, gold_set, k):
        tp = 0
        for score, e in rank_list[:k]:
            if e in gold_set:
                tp += 1
        return tp / k

    def AP(pred, gold):
        n_pred_pos = 0
        tp = 0
        sum = 0
        for score, e in pred:
            n_pred_pos += 1
            if e in gold:
                tp += 1
                sum += (tp / n_pred_pos)
        return sum / len(gold)

    #runs_list = ["pred_O_conflict_conflict"]
    runs_list = ["pred_P_match_match"]
    for run_name in runs_list:
        predictions = load_from_pickle(run_name)

        score_list_h = []
        score_list_p = []
        idx = 0
        for pred, gold in zip(predictions, gold_list):
            pred_p, pred_h = pred
            gold_p, gold_h = gold
            fail = False

            if prem_only:
                s1 = AP(pred_p, gold_p)
                if s1 < 0.96:
                    fail = True
                if fail:
                    print("-------------------")
                    print("id : ", idx)
                    print("AP : ", s1)
                    print("pred_p:", pred_p)
                    print("gold_p", gold_p)
            else:
                if gold_p:
                    s1 = p_at_k(pred_p, gold_p, 1)
                    if s1 < 0.99:
                        fail = True
                    score_list_p.append(s1)
                if gold_h :
                    s2 = p_at_k(pred_h, gold_h, 1)
                    if s2 < 0.99:
                        fail = True
                    score_list_h.append(s2)

                if fail:
                    print("-------------------")
                    print("id : ", idx)
                    print("pred_p:", pred_p)
                    print("gold_p", gold_p)
                    print("pred_h:", pred_h)
                    print("gold_h", gold_h)
            idx += 1


def run_eval():
    data_id = "conflict_0_99"
    data_id = "match"

    #runs_list = ["pred_O_conflict_conflict", "pred_deeplift_conflict", "pred_grad*input_conflict",
    #             "pred_intgrad_conflict", "pred_saliency_conflict"]

    runs_list = ["pred_P_match_match"]

    gold_list = load_gold(data_id)

    for run_name in runs_list:
        predictions = load_from_pickle(run_name)
        only_prem = False if data_id == 'conflict' else False
        scores = eval(predictions, gold_list, only_prem)
        print(run_name)
        for key in scores:
            print("{}\t{}".format(key, scores[key]))



if __name__ == '__main__':
    #run_eval()
    run_analysis()