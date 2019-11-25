from cache import load_cache

import pickle
import numpy as np
from path import output_path, data_path
import os
from data_generator import tokenizer_wo_tf
from misc_lib import right
from visualize.html_visual import get_color, Cell, HtmlVisualizer

def dev():
    train_data_feeder = load_cache("train_data_feeder")
    tokenizer = tokenizer_wo_tf.FullTokenizer(os.path.join(data_path, "bert_voca.txt"))

    html_writer = HtmlVisualizer("nli_w_dict.html", dark_mode=False)

    for _ in range(100):
        batch = train_data_feeder.get_random_batch(1)


        input_ids, input_mask, segment_ids, d_input_ids, d_input_mask, d_location_ids, y = batch

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        for i in range(len(tokens)):
            if i is not 0 and i in d_location_ids:
                tokens[i] = "<b>{}</b>".format(tokens[i])
            if tokens[i] == "[unused3]":
                tokens[i] = "[SEP]\n"

        s = tokenizer_wo_tf.pretty_tokens(tokens)
        html_writer.write_headline("Input")
        html_writer.write_paragraph(s)

        d_tokens = tokenizer.convert_ids_to_tokens(d_input_ids[0])
        for i in range(len(d_tokens)):
            if tokens[i] == "[unused5]":
                tokens[i] = "<br>\n"

        s = tokenizer_wo_tf.pretty_tokens(d_tokens)
        html_writer.write_headline("Dict def")
        html_writer.write_paragraph(s)

    html_writer.close()



if __name__ == '__main__':
    dev()

