import os
import pickle

import math

from path import output_path

f = open(os.path.join(output_path, "tlm_loss_pred.pickle"), "rb")

data = pickle.load(f)



for batch in data:
    pred_diff = batch['pred_diff']
    gold_diff = batch['gold_diff']
    per_example_loss1 = batch["per_example_loss1"]
    per_example_loss2 = batch["per_example_loss2"]
    prob1 = batch['prob1']
    prob2 = batch['prob2']
    loss_base = batch["loss_base"]
    loss_target = batch["loss_target"]

    print("Prob1\tProb2\tG)Prob_base,\tG)Prob_targ\tPDiff\tGDiff")

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for inst_idx in range(len(pred_diff)):
        num_predictions = len(pred_diff[inst_idx])
        print("Instance {}".format(inst_idx))
        for loc_idx in range(num_predictions):
            if per_example_loss1[inst_idx][loc_idx] < 1e-6 and per_example_loss2[inst_idx][loc_idx] < 1e-6:
                break

            def get(e):
                return 1-e[inst_idx][loc_idx]

            def getl(e):
                return math.exp(-e[inst_idx][loc_idx])


            print("{:04.2f}\t"
                  "{:04.2f}\t"
                  "{:04.2f}\t"
                  "{:04.2f}\t"
                  "{:04.2f}\t"
                  "{:04.2f}"
                  .format(get(prob1), get(prob2), getl(loss_base), getl(loss_target),
                         -pred_diff[inst_idx][loc_idx],
                          gold_diff[inst_idx][loc_idx]))

            pred_label = -pred_diff[inst_idx][loc_idx] > 0.2
            gold_label = gold_diff[inst_idx][loc_idx] > 0.2


            if pred_label and gold_label:
                tp += 1
            elif pred_label and not gold_label:
                fp += 1
            elif not pred_label and gold_label:
                fn += 1
            elif not pred_label and not gold_label:
                tn += 1


    acc = (tp+tn) / (tp+tn+fp+fn)
    prec = (tp) / (tp+fp)
    recl = (tp) / (tp + fn)

    f1 = 2 * (prec * recl) / (prec+recl)

    print("acc", acc)
    print("prec", prec)
    print("recall", recl)
    print("F1", f1)