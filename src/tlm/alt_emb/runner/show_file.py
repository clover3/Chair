import os

import numpy as np

from cpath import output_path
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.tokenizer_wo_tf import get_tokenizer
from tf_util.enum_features import load_record_v2
from tlm.alt_emb.prediction_analysis import get_correctness
from tlm.data_gen.feature_to_text import take
from tlm.estimator_prediction_viewer import EstimatorPredictionViewerGosford
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

def show_prediction(filename, file_path, correctness_1, correctness_2):

    data = EstimatorPredictionViewerGosford(filename)
    itr = load_record_v2(file_path)
    tokenizer = get_tokenizer()
    name = os.path.basename(filename)
    html = HtmlVisualizer(name + ".html")
    idx = 0
    for entry in data:
        features = itr.__next__()

        input_ids = entry.get_vector("input_ids")
        input_ids2 = take(features["input_ids"])
        assert np.all(input_ids == input_ids2)
        alt_emb_mask = take(features["alt_emb_mask"])
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        p_tokens, h_tokens = split_p_h_with_input_ids(tokens, input_ids)
        p_mask, h_mask = split_p_h_with_input_ids(alt_emb_mask, input_ids)

        p_cells = [Cell(p_tokens[i], 100 if p_mask[i] else 0)for i in range(len(p_tokens))]
        h_cells = [Cell(h_tokens[i], 100 if h_mask[i] else 0) for i in range(len(h_tokens))]

        label = take(features["label_ids"])[0]
        logits = entry.get_vector("logits")
        pred = np.argmax(logits)

        if not correctness_1[idx] or not correctness_2[idx]:
            html.write_paragraph("Label : {} Correct: {}/{}".format(label, correctness_1[idx], correctness_2[idx]))
            html.write_table([p_cells])
            html.write_table([h_cells])

        idx += 1

def main():
    file_path = os.path.join(output_path, "nli_tfrecord_cls_300", "dev_mis_alt_small")
    correctness_1 = get_correctness("nli_alt_emb_pred", file_path)
    correctness_2 = get_correctness("alt_emb_G100K", file_path)

    show_prediction("nli_alt_emb_pred", file_path, correctness_1, correctness_2)


if __name__ == "__main__":
    main()