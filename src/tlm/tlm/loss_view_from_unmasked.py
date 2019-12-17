import os
import pickle
from collections import Counter, defaultdict

import math
import matplotlib.pyplot as plt
import mpld3
import numpy as np

from data_generator.common import get_tokenizer
from data_generator.tokenizer_wo_tf import pretty_tokens
from path import output_path
from visualize.html_visual import Cell, HtmlVisualizer


def work():
    tokenizer = get_tokenizer()
    filename = "bert_815.pickle"
    filename = "bfn_3_200_815.pickle"
    run_name = filename[:-(len(".pickle"))]
    p = os.path.join(output_path, filename)
    data = pickle.load(open(p, "rb"))

    batch_size, seq_length = data[0]['masked_input_ids'].shape
    masked_input_ids = []
    input_ids = []
    masked_lm_example_loss = []
    masked_lm_positions = []
    for e in data[:-1]:
        masked_input_ids.append(e["masked_input_ids"])
        input_ids.append(e["input_ids"])
        masked_lm_example_loss.append(np.reshape(e["masked_lm_example_loss"], [batch_size, -1]))
        masked_lm_positions.append(e["masked_lm_positions"])

    input_ids = np.concatenate(input_ids)
    masked_input_ids = np.concatenate(masked_input_ids)
    masked_lm_example_loss = np.concatenate(masked_lm_example_loss)
    masked_lm_positions = np.concatenate(masked_lm_positions)

    html_writer = HtmlVisualizer(run_name + ".html", dark_mode=False)
    n_instance = len(input_ids)
    for inst_idx in range(200):

        tokens = tokenizer.convert_ids_to_tokens(masked_input_ids[inst_idx])
        ans_tokens = tokenizer.convert_ids_to_tokens(input_ids[inst_idx])

        loss_at_loc = {p:l for l, p in zip(masked_lm_example_loss[inst_idx], masked_lm_positions[inst_idx])}


        cells = []
        for i in range(len(tokens)):
            score = 0
            if tokens[i] == "[MASK]":
                tokens[i] = "[{}]".format(ans_tokens[i])
                score = loss_at_loc[i] * 255 / 25
            if tokens[i] == "[SEP]":
                tokens[i] = "[SEP]<br>"


            if tokens[i] != "[PAD]":
                cells.append(Cell(tokens[i], score))
        #s = tokenization.pretty_tokens(tokens)

        rows = []
        row = []
        for cell in cells:
            row.append(cell)
            if len(row) == 20:
                html_writer.write_table([row])
                row = []



        loss_infos = []
        for loss, pos in zip(masked_lm_example_loss[inst_idx], masked_lm_positions[inst_idx]):
            loss_infos.append((loss, pos))

        loss_infos.sort(key= lambda x:x[1])

        rows = []
        for loss, pos in loss_infos:
            rows.append((Cell(pos), Cell(loss)))

        html_writer.write_table(rows)

    html_writer.close()


def diff_view():
    tokenizer = get_tokenizer()
    filename = "bert_815.pickle"
    p = os.path.join(output_path, filename)
    data = pickle.load(open(p, "rb"))
    filename = "bfn_3_200_815.pickle"
    p = os.path.join(output_path, filename)
    data2 = pickle.load(open(p, "rb"))

    run_name = "diff"



    batch_size, seq_length = data[0]['masked_input_ids'].shape
    masked_input_ids = []
    input_ids = []
    masked_lm_example_loss = []

    masked_lm_positions = []
    masked_lm_ids = []
    for e in data[:-1]:
        masked_input_ids.append(e["masked_input_ids"])
        input_ids.append(e["input_ids"])
        masked_lm_example_loss.append(np.reshape(e["masked_lm_example_loss"], [batch_size, -1]))
        masked_lm_positions.append(e["masked_lm_positions"])
        masked_lm_ids.append(e["masked_lm_ids"])

    masked_lm_example_loss2 = []
    for e in data2[:-1]:
        masked_lm_example_loss2.append(np.reshape(e["masked_lm_example_loss"], [batch_size, -1]))

    masked_lm_example_loss2 = np.concatenate(masked_lm_example_loss2)


    input_ids = np.concatenate(input_ids)
    masked_input_ids = np.concatenate(masked_input_ids)
    masked_lm_example_loss = np.concatenate(masked_lm_example_loss)
    masked_lm_positions = np.concatenate(masked_lm_positions)
    masked_lm_ids = np.concatenate(masked_lm_ids)

    html_writer = HtmlVisualizer(run_name + ".html", dark_mode=False)
    n_instance = len(input_ids)
    for inst_idx in range(n_instance):

        tokens = tokenizer.convert_ids_to_tokens(masked_input_ids[inst_idx])
        ans_tokens = tokenizer.convert_ids_to_tokens(input_ids[inst_idx])

        ans_keys = dict(zip(masked_lm_positions[inst_idx], tokenizer.convert_ids_to_tokens(masked_lm_ids[inst_idx])))

        loss_at_loc = {p:l for l, p in zip(masked_lm_example_loss[inst_idx], masked_lm_positions[inst_idx])}
        loss_at_loc2 = {p:l for l, p in zip(masked_lm_example_loss2[inst_idx], masked_lm_positions[inst_idx])}

        score_at_loc = {k: math.exp(-v) for k,v in loss_at_loc.items()}
        score_at_loc2 = {k: math.exp(-v) for k,v in loss_at_loc2.items()}

        def is_dependent(token):
            return len(token) == 1 and not token[0].isalnum()

        cells = []
        for i in range(len(tokens)):
            f_inverse = False
            score = 0
            if tokens[i] == "[MASK]" or i in loss_at_loc:
                tokens[i] = "[{}-{}]".format(i, ans_keys[i])
                score = (score_at_loc2[i] - score_at_loc[i]) * 180
                score = -score
                if score < 0:
                    f_inverse = True
                    score = abs(score)
            if tokens[i] == "[SEP]":
                tokens[i] = "[SEP]<br>"


            if tokens[i] != "[PAD]":
                term = tokens[i]
                cont_left = term[:2] == "##"
                cont_right = i+1 < len(tokens) and tokens[i+1][:2] == "##"
                if i+1 < len(tokens):
                    dependent_right = is_dependent(tokens[i+1])
                else:
                    dependent_right = False

                dependent_left = is_dependent(tokens[i])

                if cont_left:
                    term = term[2:]

                space_left = "&nbsp;" if not (cont_left or dependent_left) else ""
                space_right = "&nbsp;" if not (cont_right or dependent_right) else ""

                if not f_inverse:
                    cells.append(Cell(term, score, space_left, space_right))
                else:
                    cells.append(Cell(term, score, space_left, space_right, target_color="R"))
        #s = tokenization.pretty_tokens(tokens)

        rows = []
        row = []
        for cell in cells:
            row.append(cell)
            if len(row) == 20:
                html_writer.write_table([row])
                row = []



        loss_infos = []
        for loss, pos in zip(masked_lm_example_loss[inst_idx], masked_lm_positions[inst_idx]):
            loss_infos.append((loss, pos))

        loss_infos.sort(key= lambda x:x[1])

        rows = []
        for loss, pos in loss_infos:
            loss1 = score_at_loc[pos]
            loss2 = score_at_loc2[pos]
            loss_diff = loss1 - loss2
            rows.append((Cell(pos), Cell(loss1), Cell(loss2), Cell(loss_diff)))

        html_writer.write_table(rows)

    html_writer.close()



def pred_loss_view():
    tokenizer = get_tokenizer()
    filename = "tlm_loss_pred.pickle"
    filename = "tlm_loss_pred_on_dev.pickle"
    p = os.path.join(output_path, filename)
    data = pickle.load(open(p, "rb"))

    batch_size, seq_length = data[0]['input_ids'].shape

    keys = list(data[0].keys())
    vectors = {}


    for e in data:
        for key in keys:
            if key not in vectors:
                vectors[key] = []
            vectors[key].append(e[key])

    for key in keys:
        vectors[key] = np.concatenate(vectors[key], axis=0)


    html_writer = HtmlVisualizer("pred_make_sense_dev.html", dark_mode=False)

    n_instance = len(vectors['input_ids'])
    n_instance = min(n_instance, 100)
    for inst_idx in range(n_instance):
        tokens = tokenizer.convert_ids_to_tokens(vectors['input_ids'][inst_idx])
        locations = list(vectors['masked_lm_positions'][inst_idx])

        def is_dependent(token):
            return len(token) == 1 and not token[0].isalnum()

        cells = []
        for i in range(len(tokens)):
            f_same_pred = False
            score = 0
            if i in locations and i != 0:
                i_idx = locations.index(i)
                tokens[i] = "[{}:{}]".format(i_idx, tokens[i])
                pred_diff = vectors['pred_diff'][inst_idx][i_idx]
                gold_diff = vectors['gold_diff'][inst_idx][i_idx]
                pred_label = pred_diff > 0.3
                gold_label = gold_diff > 0.3
                if pred_label:
                    score = 100
                    if gold_label:
                        f_same_pred = True
                else:
                    if gold_label:
                        score = 30
                        f_same_pred = False

            if tokens[i] == "[SEP]":
                tokens[i] = "[SEP]<br>"

            if tokens[i] != "[PAD]":
                term = tokens[i]
                cont_left = term[:2] == "##"
                cont_right = i+1 < len(tokens) and tokens[i+1][:2] == "##"
                if i+1 < len(tokens):
                    dependent_right = is_dependent(tokens[i+1])
                else:
                    dependent_right = False

                dependent_left = is_dependent(tokens[i])

                if cont_left:
                    term = term[2:]

                space_left = "&nbsp;" if not (cont_left or dependent_left) else ""
                space_right = "&nbsp;" if not (cont_right or dependent_right) else ""

                if f_same_pred:
                    cells.append(Cell(term, score, space_left, space_right))
                else:
                    cells.append(Cell(term, score, space_left, space_right, target_color="R"))

        row = []
        for cell in cells:
            row.append(cell)
            if len(row) == 20:
                html_writer.write_table([row])
                row = []

        row_head = [Cell("Index"),
                    Cell("P]Prob1"), Cell("P]Prob2"),
                    Cell("G]Prob1"), Cell("G]Prob2"),
                    Cell("P]Diff"), Cell("G]Diff"),
                    ]
        
        def f_cell(obj):
            return Cell("{:04.2f}".format(obj))
            
        rows = [row_head]
        pred_diff_list = []
        gold_diff_list = []
        for idx, pos in enumerate(locations):
            if pos == 0:
                break
            pred_diff = vectors['pred_diff'][inst_idx][idx]
            gold_diff = vectors['gold_diff'][inst_idx][idx]
            pred_diff_list.append(pred_diff)
            gold_diff_list.append(gold_diff)

            row = [Cell(idx),
                   f_cell(vectors['prob1'][inst_idx][idx]),
                   f_cell(vectors['prob2'][inst_idx][idx]),
                   f_cell(math.exp(-vectors['loss_base'][inst_idx][idx])),
                   f_cell(math.exp(-vectors['loss_target'][inst_idx][idx])),
                   f_cell(pred_diff),
                   f_cell(gold_diff),
                   ]
            rows.append(row)

        html_writer.write_table(rows)

        pred_diff = np.average(pred_diff_list)
        gold_diff = np.average(gold_diff_list)
        html_writer.write_paragraph("Average Pred diff ={:04.2f} Observed diff={:04.2f} ".format(pred_diff, gold_diff))

        if pred_diff > 0.3 :
            html_writer.write_headline("High Drop")
        elif pred_diff < 0.1 :
            html_writer.write_headline("Low Drop")


def loss_drop_tendency():
    tokenizer = get_tokenizer()
    filename = "tlm_loss_pred_on_dev.pickle"
    filename = "412.pickle"
    p = os.path.join(output_path, filename)
    data = pickle.load(open(p, "rb"))

    batch_size, seq_length = data[0]['input_ids'].shape

    keys = list(data[0].keys())
    vectors = {}

    for e in data:
        for key in keys:
            if key not in vectors:
                vectors[key] = []
            vectors[key].append(e[key])

    for key in keys:
        vectors[key] = np.concatenate(vectors[key], axis=0)


    n_instance = len(vectors['input_ids'])
    print("n_instance ", n_instance )
    token_cnt = Counter()
    acc_prob_before = Counter()
    acc_prob_after = Counter()
    num_predictions = len(vectors["grouped_positions"][0][0])

    prev_word = defaultdict(list)
    context = defaultdict(list)

    def combine(t1,t2):
        if t2.startswith("##"):
            return t1+t2
        else:
            return t1 + "_" + t2

    n_instance = 1000
    for i in range(n_instance):
        tokens = tokenizer.convert_ids_to_tokens(vectors['input_ids'][i])
        positions = vectors["grouped_positions"][i]

        num_trials = len(positions)
        for t_i in range(num_trials):
            for p_i in range(num_predictions):
                loc = vectors["grouped_positions"][i][t_i][p_i]
                loss1 = vectors["grouped_loss1"][i][t_i][p_i]
                loss2 = vectors["grouped_loss2"][i][t_i][p_i]

                t = combine(tokens[loc-1], tokens[loc])
                prev_word[t].append(tokens[loc-1])

                ctx = pretty_tokens(tokens[loc-5:loc+4], drop_sharp=False)
                context[t].append(ctx)

                prob_before = math.exp(-loss1)
                prob_after = math.exp(-loss2)
                token_cnt[t] += 1
                acc_prob_before[t] += prob_before
                acc_prob_after[t] += prob_after

    infos = []

    for t in token_cnt:
        cnt = token_cnt[t]
        avg_prob_before = acc_prob_before[t] / cnt
        avg_prob_after = acc_prob_after[t] / cnt
        avg_diff = avg_prob_before - avg_prob_after
        e = t, avg_prob_before, avg_prob_after, avg_diff, cnt
        infos.append(e)

    infos = list([e for e in infos if e[4] > 10])


    def entropy(cnt_dict:Counter):
        total = sum(cnt_dict.values())

        ent = 0
        for key, value in cnt_dict.items():
            p = value / total

            ent += -p * math.log(p)
        return ent

    def print_n(e_list, n):
        for e in e_list[:n]:
            t, avg_prob_before, avg_prob_after, avg_diff, cnt = e
            print("{}  ({})".format(t, cnt))
            print("Before : {0:3f}".format(avg_prob_before))
            print("After  : {0:3f}".format(avg_prob_after))
            print("AvgDiff: {0:3f}".format(avg_diff))
            term_stat = Counter(prev_word[e[0]])
            print(term_stat)
            print(context[t])
            print("Entropy: ", entropy(term_stat))

    print(type(infos[0][0]))
    print(type(infos[0][1]))
    print(type(infos[0][2]))


    print("<< Most common >>")
    infos.sort(key=lambda x:x[1], reverse=True)
    print_n(infos, 10)
    print("---------------------")

    infos.sort(key=lambda x: x[3], reverse=True)
    print("<<  Big Drop  >>")
    print_n(infos, 10)
    print("---------------------")

    infos.sort(key=lambda x: x[3], reverse=False)
    print("<< Negative Drop (NLI Improve >>")
    print_n(infos, 10)
    print("---------------------")

    plt.rcParams.update({'font.size': 22})
    fig, ax = plt.subplots()
    infos = list([e for e in infos if e[4] > 30])

    y = list([x[1] for x in infos])
    z = list([x[2] for x in infos])
    fig.set_size_inches(18.5, 10.5)

    ax.scatter(z, y)
    x = np.linspace(0, 1, 1000)
    ax.plot(x, x)

    for i, e in enumerate(infos):
        ax.annotate(e[0], (z[i], y[i]))

    mpld3.show()

def debug_it():
    y = [0] *10
    z = [0] * 10
    fig, ax = plt.subplots()

    infos = [["good"]] * 10
    ax.scatter(z, y)

    for i, e in enumerate(infos):
        ax.annotate(e[0], (z[i], y[i]))

    mpld3.show()




if __name__ == '__main__':
    loss_drop_tendency()


