import sys

from data_generator.tokenizer_wo_tf import get_tokenizer
from tf_util.enum_features import load_record
from tlm.data_gen.feature_to_text import take
from visualize.html_visual import HtmlVisualizer, Cell


def show_feature_text(tfrecord_path, output_file_name):
    html = HtmlVisualizer(output_file_name)
    tokenizer = get_tokenizer()

    for feature in load_record(tfrecord_path):
        write_feature_to_html(feature, html, tokenizer)


def write_feature_to_html(feature, html, tokenizer):
    input_ids = take(feature['input_ids'])
    focus_msak = take(feature['focus_mask'])
    label_ids = take(feature['label_ids'])
    text1 = tokenizer.convert_ids_to_tokens(input_ids)


    row = []
    for i in range(len(input_ids)):
        highlight_score = 100 if focus_msak[i] else 0
        row.append(Cell(text1[i], highlight_score))

    html.write_headline("{}".format(label_ids[0]))
    html.multirow_print(row)


if __name__ == "__main__":
    tfrecord_path = sys.argv[1]
    output_file_name = sys.argv[2]
    show_feature_text(tfrecord_path, output_file_name)