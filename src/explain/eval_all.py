
from evaluation import *
from cache import *
from data_generator.NLI.nli import *
import path

def eval(pred_list, gold_list, only_prem, only_hypo = False):
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
            "MAP":MAP_p,
            "AUC": p_auc,
        }
        return scores
    elif only_hypo:
        p1_p, p1_h = p_at_k_list_ind(pred_list, gold_list, 1)
        # p_at_20 = p_at_k_list(pred_list, gold_list, 20)
        p_auc, h_auc = PR_AUC_ind(pred_list, gold_list)
        MAP_p, MAP_h = MAP_ind(pred_list, gold_list)
        scores = {
            "P@1": p1_h,
            "MAP": MAP_h,
            "AUC": h_auc,
        }
        return scores
    else:
        p1 = p_at_k_list(pred_list, gold_list, 1)
        # p_at_20 = p_at_k_list(pred_list, gold_list, 20)
        auc_score = PR_AUC(pred_list, gold_list)
        MAP_score = MAP(pred_list, gold_list)
        scores = {
            "P@1": p1,
            "MAP": MAP_score,
            "AUC": auc_score,
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
    elif data_id.startswith("test_"):
        data = load_nli_explain_3(data_id + "_idx", data_id)
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


def load_prediction(name):
    file_path = os.path.join(path.output_path, "prediction", "nli", name + ".pickle")
    return pickle.load(open(file_path, "rb"))





def paired_p_test(scorer, predictions1, predictions2, gold_list, only_prem, only_hypo):
    from scipy import stats

    p_score_list1, h_score_list1 = scorer(predictions1, gold_list)
    p_score_list2, h_score_list2 = scorer(predictions2, gold_list)

    assert len(p_score_list1) == len(p_score_list2)
    assert len(h_score_list1) == len(h_score_list2)

    if only_prem:
        score_list_1 = p_score_list1
        score_list_2 = p_score_list2
    elif only_hypo:
        score_list_1 = h_score_list1
        score_list_2 = h_score_list2
    else:
        score_list_1 = p_score_list1 + h_score_list1
        score_list_2 = p_score_list2 + h_score_list2

    _, p = stats.ttest_rel(score_list_1, score_list_2)
    return p



def paired_p_test_runner():
    best_runner = {
            'match':['Y_match', 'deletion'],
            'mismatch':['V_mismatch','saliency'],
            'conflict':['Y_conflict', 'deletion'],
        }

    for target_label in ["mismatch", "match", "conflict"]:
        data_id = "test_{}".format(target_label)
        gold_list = load_gold(data_id)

        only_prem = True if target_label == 'match' else False
        only_hypo = True if target_label == 'mismatch' else False

        predictions_list = []
        for method_name in best_runner[target_label]:
            run_name = "pred_" + method_name + "_" + data_id
            predictions_list.append(load_prediction(run_name))

        def p_at_1(pred, gold):
            return p_at_k_list_inner(pred, gold, 1)

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

        print(best_runner[target_label])
        p_pat1 = paired_p_test(p_at_1, predictions_list[0], predictions_list[1], gold_list, only_prem, only_hypo)
        print("p-value for p@1", p_pat1)
        p_AP = paired_p_test(MAP_inner, predictions_list[0], predictions_list[1], gold_list, only_prem, only_hypo)
        print("p-value for MAP", p_AP)



def mismatch_p():
    methods = ['V_mismatch', 'W_mismatch']
    target_label = "mismatch"
    data_id = "test_{}".format(target_label)
    gold_list = load_gold(data_id)

    only_prem = True if target_label == 'match' else False
    only_hypo = True if target_label == 'mismatch' else False

    predictions_list = []
    for method_name in methods:
        run_name = "pred_" + method_name + "_" + data_id
        predictions_list.append(load_prediction(run_name))

    print(methods)
    p_AP = paired_p_test(MAP_inner, predictions_list[0], predictions_list[1], gold_list, only_prem, only_hypo)
    print("p-value for MAP", p_AP)


def run_test_eval():
    target_label =  "mismatch"
    data_id = "test_{}".format(target_label)
    label_name = "test_{}_idx".format(target_label)
    gold_list = load_gold(data_id)

    model_name = {
        'match':'Y_match',
        'mismatch':'V_mismatch',
        'conflict':'Y_conflict',
    }[target_label]

    run_names = []
    for method_name in ["random", "idf", "saliency",  "grad*input", "intgrad", "deletion", "deletion_seq", model_name, 'W_mismatch']:
        run_name = "pred_" + method_name + "_" + data_id
        run_names.append(run_name)

    for run_name in run_names:
        predictions = load_prediction(run_name)
        print(run_name)
        if target_label =='conflict':
            scores = eval(predictions, gold_list, False, False)
        elif target_label == 'match':
            scores = eval(predictions, gold_list, True, False)
        elif target_label == 'mismatch':
            scores = eval(predictions, gold_list, False, True)
        for key in scores:
            print("{}".format(key), end="\t")
        print()
        for key in scores:
            print("{}".format(scores[key]), end="\t")
        print()

if __name__ == '__main__':
    #run_eval()
    #run_test_eval()
    mismatch_p()
    #paired_p_test_runner()