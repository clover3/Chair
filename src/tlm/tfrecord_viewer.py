import collections
import os
import sys

from data_generator import tokenizer_wo_tf
from path import data_path
from tf_util.enum_features import load_record
from tlm.token_utils import cells_from_tokens
from visualize.html_visual import HtmlVisualizer


def repack_features(feature):
    new_feature = collections.OrderedDict()
    for key in feature.keys():
        new_feature[key] = feature[key]
    return new_feature


def inst2str(feature, tokenizer):
    def pretty(t):
        if t[:2] == "##":
            return t[2:]
        else:
            return t

    a = feature["input_ids"].int64_list.value

    mask_idx = 0
    mask_tokens = tokenizer.convert_ids_to_tokens(feature["masked_lm_ids"].int64_list.value)
    masked_positions = feature["masked_lm_positions"].int64_list.value
    out_str = ""
    for i, t in enumerate(tokenizer.convert_ids_to_tokens(a)):
        if t == "[PAD]":
            break

        skip_space = t[:2] == "##"
        if not skip_space:
            out_str += " "

        t = pretty(t)
        if i in masked_positions:
            out_str += "({}={})".format(t, mask_tokens[mask_idx])
            mask_idx += 1
        else:
            out_str += t
    return out_str


def print_as_html(fn):
    examples = load_record(fn)
    tokenizer = tokenizer_wo_tf.FullTokenizer(os.path.join(data_path, "bert_voca.txt"))

    html_output = HtmlVisualizer("out_name.html")

    for feature in examples:
        masked_inputs = feature["input_ids"].int64_list.value
        idx = 0
        step = 512
        while idx < len(masked_inputs):
            slice = masked_inputs[idx:idx+step]
            tokens = tokenizer.convert_ids_to_tokens(slice)
            idx += step
            cells = cells_from_tokens(tokens)
            html_output.multirow_print(cells)
        html_output.write_paragraph("----------")




def read(fn):
    examples = load_record(fn)
    tokenizer = tokenizer_wo_tf.FullTokenizer(os.path.join(data_path, "bert_voca.txt"))

    for feature in examples:
        print(inst2str(feature, tokenizer))
        print()
        print()


if __name__ == "__main__":
    fn = sys.argv[1]
    print_as_html(fn)