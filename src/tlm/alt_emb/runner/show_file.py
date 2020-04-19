import os

from cpath import output_path
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.common import get_tokenizer
from tf_util.enum_features import load_record_v2
from tlm.data_gen.feature_to_text import take
from visualize.html_visual import HtmlVisualizer, Cell


def show_tfrecord(file_path):

    itr = load_record_v2(file_path)
    tokenizer = get_tokenizer()
    name = os.path.basename(file_path)
    html = HtmlVisualizer(name + ".html")
    for features in itr:
        input_ids = take(features["input_ids"])
        alt_emb_mask = take(features["alt_emb_mask"])
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        p_tokens, h_tokens = split_p_h_with_input_ids(tokens, input_ids)
        p_mask, h_mask = split_p_h_with_input_ids(alt_emb_mask, input_ids)


        p_cells = [Cell(p_tokens[i], 100 if p_mask[i] else 0)for i in range(len(p_tokens))]
        h_cells = [Cell(h_tokens[i], 100 if h_mask[i] else 0) for i in range(len(h_tokens))]

        label = take(features["label_ids"])[0]

        html.write_paragraph("Label : {}".format(label))
        html.write_table([p_cells])
        html.write_table([h_cells])


def main():
    file_path = os.path.join(output_path, "nli_tfrecord_cls_300", "dev_mis_alt_small")
    show_tfrecord(file_path)


if __name__ == "__main__":
    main()