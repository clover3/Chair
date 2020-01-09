import os
import pickle

import math
import numpy as np
import scipy.special

from cpath import output_path
from data_generator.common import get_tokenizer
from misc_lib import lmap
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer
from visualize.html_visual import HtmlVisualizer, Cell


def probabilty(scores, amp):
    prob = scipy.special.softmax(scores * amp)
    return prob


def draw(in_file, out_file):

    filename = os.path.join(output_path, in_file)
    data = EstimatorPredictionViewer(filename)
    amp = 10
    html_writer = HtmlVisualizer(out_file, dark_mode=False)

    tokenizer = get_tokenizer()
    for inst_i, entry in enumerate(data):
        if inst_i > 100:
            break

        input_ids = entry.get_vector("input_ids")
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        #tokens = entry.get_tokens("input_ids")
        prob1 = entry.get_vector("prob1")
        prob2 = entry.get_vector("prob2")

        proximity = (1-(prob1 - prob2))
        difficulty = np.power(1-prob1, 0.3)
        scores = proximity * difficulty

        prob_scores = probabilty(scores, amp)
        prob_strs = ["{:06.6f}".format(v*1000) for v in prob_scores]

        def normalize(prob):
            # 0-> Good
            # -1 -> Bad
            return prob * 1000 * 25

        norm_scores = lmap(normalize, prob_scores)


        cells = data.cells_from_tokens(tokens, norm_scores, stop_at_pad=False)
        cells2 = data.cells_from_anything(prob1, norm_scores)
        cells3 = data.cells_from_anything(prob2, norm_scores)
        cells31 = data.cells_from_anything(difficulty, norm_scores)
        cells4 = data.cells_from_anything(scores, norm_scores)
        cells5 = data.cells_from_anything(prob_strs, norm_scores)

        row1 = [Cell("TEXT:")]
        row2 = [Cell("prob1:")]
        row3 = [Cell("prob2:")]
        row31 = [Cell("difficulty:")]
        row4 = [Cell("priority:")]
        row5 = [Cell("Mask Prob")]

        for idx, cell in enumerate(cells):
            row1.append(cell)
            row2.append(cells2[idx])
            row3.append(cells3[idx])
            row31.append(cells31[idx])
            row4.append(cells4[idx])
            row5.append(cells5[idx])

            if len(row1) == 20:
                html_writer.write_table([row1, row2, row3, row31, row4, row5])

                row1 = [Cell("TEXT:")]
                row2 = [Cell("prob1:")]
                row3 = [Cell("prob2:")]
                row31 = [Cell("difficulty:")]
                row4 = [Cell("priority:")]
                row5 = [Cell("Mask Prob")]


def draw2(in_file, out_file):
    filename = os.path.join(output_path, in_file)
    data = EstimatorPredictionViewer(filename)
    html_writer = HtmlVisualizer(out_file, dark_mode=False)

    tokenizer = get_tokenizer()
    for inst_i, entry in enumerate(data):
        if inst_i > 100:
            break

        tokens = entry.get_tokens("input_ids")
        # tokens = entry.get_tokens("input_ids")
        prob1 = entry.get_vector("prob1")
        prob2 = entry.get_vector("prob2")
        real_loss1 = entry.get_vector("per_example_loss1")
        real_loss2 = entry.get_vector("per_example_loss2")

        masked_lm_positions = entry.get_vector("masked_lm_positions")

        for i, loc in enumerate(masked_lm_positions):

            tokens[loc] = "[{}:{}]".format(i, tokens[loc])

        html_writer.multirow_print(data.cells_from_tokens(tokens))

        row2 = [Cell("prob1:")] + data.cells_from_anything(prob1)
        row3 = [Cell("prob2:")] + data.cells_from_anything(prob2)
        row4 = [Cell("real_loss1:")] + data.cells_from_anything(real_loss1)
        row5 = [Cell("real_loss2:")] + data.cells_from_anything(real_loss2)
        html_writer.multirow_print_from_cells_list([row2, row3, row4,row5])

def plain_analyze():
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


if __name__ == "__main__":
    "tlm_scores.pickle"
    in_path = "loss_predictor4.pickle"
    out_path = "loss_predictor4.html"
    draw2(in_path, out_path)