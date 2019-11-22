import pickle
import numpy as np
from path import output_path, data_path
import os
from misc_lib import right
from scipy.stats import ttest_ind
import tensorflow as tf
from data_generator import tokenizer_wo_tf
from visualize.html_visual import get_color, Cell, HtmlVisualizer

def load_record(fn):
    for record in tf.python_io.tf_record_iterator(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        yield feature


def load():
    tokenizer = tokenizer_wo_tf.FullTokenizer(os.path.join(data_path, "bert_voca.txt"))

    data_id = "2"

    n_list = open(os.path.join(output_path, "lookup_n", data_id), "r").readlines()
    p = os.path.join(output_path, "example_loss.pickle")
    data = pickle.load(open(p, "rb"))
    data = data[0]["masked_lm_example_loss"]

    feature_itr = load_record(os.path.join(output_path, "lookup_example", data_id))

    def take(v):
        return v.int64_list.value

    n = len(n_list)
    feature_idx = 0
    html_writer = HtmlVisualizer("lookup_loss2.html", dark_mode=False)

    for i in range(n):
        n_sample = int(n_list[i])
        rows = []
        assert n_sample > 0
        for j in range(n_sample):
            feature = feature_itr.__next__()

            input_ids = take(feature["input_ids"])
            masked_lm_ids = take(feature["masked_lm_ids"])
            masked_lm_positions = take(feature["masked_lm_positions"])
            input_mask = take(feature["input_mask"])
            selected_word = take(feature["selected_word"])
            d_input_ids = take(feature["d_input_ids"])
            d_location_ids = take(feature["d_location_ids"])

            word_tokens = tokenizer.convert_ids_to_tokens(selected_word)
            word = tokenizer_wo_tf.pretty_tokens((word_tokens))

            emph_word = "<b>" + word + "</b>"

            if j ==0 :
                mask_ans = {}
                masked_terms = tokenizer.convert_ids_to_tokens(masked_lm_ids)
                for pos, id in zip(list(masked_lm_positions), masked_terms):
                    mask_ans[pos] = id

                tokens = tokenizer.convert_ids_to_tokens(input_ids)

            for i in range(len(tokens)):
                if tokens[i] == "[MASK]":
                    tokens[i] = "[MASK_{}: {}]".format(i, mask_ans[i])
                if i in d_location_ids and i is not 0:
                    if tokens[i - 1] != emph_word:
                        tokens[i] = emph_word
                    else:
                        tokens[i] = "-"

            def_str = tokenizer_wo_tf.pretty_tokens(tokenizer.convert_ids_to_tokens(d_input_ids), True)
            row = list()
            row.append(Cell(word))
            row.append(Cell(data[feature_idx]))
            row.append(Cell(def_str))
            rows.append(row)

            feature_idx += 1

        s = tokenizer_wo_tf.pretty_tokens(tokens, True)
        html_writer.write_paragraph(s)

        html_writer.write_table(rows)

    html_writer.close()


if __name__ == '__main__':
    load()

