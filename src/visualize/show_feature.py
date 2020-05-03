import sys

from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.common import get_tokenizer
from data_generator.tokenizer_wo_tf import pretty_tokens
from tf_util.enum_features import load_record
from tlm.data_gen.feature_to_text import take
from visualize.html_visual import HtmlVisualizer


def show_feature_text(tfrecord_path, output_file_name):
    html = HtmlVisualizer(output_file_name)
    tokenizer = get_tokenizer()

    for feature in load_record(tfrecord_path):
        write_feature_to_html(feature, html, tokenizer)


def write_feature_to_html(feature, html, tokenizer):
    input_ids = take(feature['input_ids'])
    label_ids = take(feature['label_ids'])
    seg1, seg2 = split_p_h_with_input_ids(input_ids, input_ids)
    text1 = tokenizer.convert_ids_to_tokens(seg1)
    text2 = tokenizer.convert_ids_to_tokens(seg2)
    text1 = pretty_tokens(text1, True)
    text2 = pretty_tokens(text2, True)
    html.write_headline("{}".format(label_ids[0]))
    html.write_paragraph(text1)
    html.write_paragraph(text2)


if __name__ == "__main__":
    tfrecord_path = sys.argv[1]
    output_file_name = sys.argv[2]
    show_feature_text(tfrecord_path, output_file_name)