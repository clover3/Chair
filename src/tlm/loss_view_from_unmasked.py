import os
import pickle

import numpy as np

from data_generator.common import get_tokenizer
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


        cells = []
        for i in range(len(tokens)):
            f_inverse = False
            score = 0
            if tokens[i] == "[MASK]":
                tokens[i] = "[{}]".format(ans_keys[i])
                score = (loss_at_loc2[i] - loss_at_loc[i]) * 255 / 25
                if score < 0:
                    f_inverse = True
                    score = abs(score)
            if tokens[i] == "[SEP]":
                tokens[i] = "[SEP]<br>"


            if tokens[i] != "[PAD]":
                if not f_inverse:
                    cells.append(Cell(tokens[i], score))
                else:
                    cells.append(Cell(tokens[i], score, target_color="R"))
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


if __name__ == '__main__':
    diff_view()

