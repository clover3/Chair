import numpy as np
import scipy.special

from contradiction.medical_claims.token_tagging.visualizer.deletion_score_to_html import make_nli_prediction_summary_str
from cpath import pjoin, data_path
from data_generator.bert_input_splitter import get_sep_loc
from data_generator.tokenizer_wo_tf import EncoderUnitPlain, get_tokenizer
from explain.bert_components.cls_probe_base_visualize import layer_no_to_name, get_row_cells
from explain.bert_components.cmd_nli import ModelConfig
from explain.bert_components.load_probe import load_probe
from tlm.token_utils import cells_from_tokens
from trainer.np_modules import get_batches_ex
from visualize.html_visual import HtmlVisualizer, Cell


def write_html(html, input_ids, logits, probe_logits, y):
    num_layers = 12 + 1
    print(len(probe_logits))
    print(probe_logits[0].shape)
    tokenizer = get_tokenizer()
    num_data = len(input_ids)
    probs_arr = scipy.special.softmax(logits, axis=-1)
    for data_idx in range(num_data)[:100]:
        sep1_idx, sep2_idx = get_sep_loc(input_ids[data_idx])
        tokens = tokenizer.convert_ids_to_tokens(input_ids[data_idx])
        first_padding_loc = tokens.index("[PAD]")
        display_len = first_padding_loc + 1
        pred_str = make_nli_prediction_summary_str(probs_arr[data_idx])
        html.write_paragraph("Prediction: {}".format(pred_str))
        html.write_paragraph("gold label={}".format(y[data_idx]))
        text_rows = [Cell("")] + cells_from_tokens(tokens)
        rows = [text_rows]
        mid_pred_rows = []
        for layer_no in range(num_layers):
            layer_logit = probe_logits[layer_no][data_idx]
            probs = scipy.special.softmax(layer_logit, axis=1)
            head = Cell(layer_no_to_name(layer_no))
            row = get_row_cells(head, probs)
            mid_pred_rows.append(row)

        head = Cell("wavg")
        hidden_layers_logits = np.array([probe_logits[i][data_idx] for i in range(1, 13)])
        print('hidden_layers_logits', hidden_layers_logits.shape)
        all_layer_probes = scipy.special.softmax(hidden_layers_logits, axis=2)
        weights = np.array([13 - i for i in range(1, 13)])
        weights_e = np.expand_dims(np.expand_dims(weights, 1), 2)
        avg_probs = np.sum(all_layer_probes * weights_e, axis=0) / sum(weights)
        row = get_row_cells(head, avg_probs)
        mid_pred_rows.append(row)

        rows.extend(mid_pred_rows[::-1])

        rows = [row[:display_len] for row in rows]
        html.write_table(rows)

        html.f_html.flush()

def main():
    model_config = ModelConfig()
    voca_path = pjoin(data_path, "bert_voca.txt")
    tokenizer = get_tokenizer()
    d_encoder = EncoderUnitPlain(model_config.max_seq_length, voca_path)
    def encode(p_text, h_text):
        p_tokens_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(p_text))
        h_tokens_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(h_text))
        d = d_encoder.encode_inner(p_tokens_id, h_tokens_id)
        p = d["input_ids"], d["input_mask"], d["segment_ids"]
        return p

    model, bert_cls_probe = load_probe(model_config)
    save_name = "cls_probe_console.html"
    while True:
        sent1 = input("Premise: ")
        sent2 = input("Hypothesis: ")
        single_x = encode(sent1, sent2)
        X = get_batches_ex([single_x], 1, 3)[0]
        html = HtmlVisualizer(save_name)
        logits, probes = bert_cls_probe(X)
        write_html(html, X[0], logits, probes, [0])



if __name__ == "__main__":
    main()


