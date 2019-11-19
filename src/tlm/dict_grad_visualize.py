import pickle
import numpy as np
from path import output_path, data_path
import os
from data_generator import tokenizer_wo_tf


def load_and_analyze_gradient():
    p = os.path.join(output_path, "dict_grad.pickle")
    data = pickle.load(open(p, "rb"))
    data = data[0]
    tokenizer = tokenizer_wo_tf.FullTokenizer(os.path.join(data_path, "bert_voca.txt"))

    analyze_gradient(data, tokenizer)

def parse_data(data):
    data = data[0]
    gradients = data['gradients']
    d_input_ids = data['d_input_ids']
    input_ids = data['masked_input_ids']
    masked_lm_positions = data["masked_lm_positions"]
    return input_ids, d_input_ids, gradients, masked_lm_positions


def reshape_gradienet(gradients, n_inst, def_len, hidden_dim):
    r = np.reshape(gradients, [-1, n_inst, def_len, hidden_dim])
    return np.transpose(r, [1,0,2,3])


def normalize255(v, max):
    if max==0:
        return 0
    return v/max * 255

def get_color(r):
    r = 255 - int(r)
    bg_color = ("%02x" % r) + ("%02x" % r) + "ff"
    return bg_color


def analyze_gradient(data, tokenizer):
    gradients = data['gradients']
    input_ids = data['input_ids']
    d_input_ids = data['d_input_ids']
    mask_input_ids = data['masked_input_ids']
    masked_lm_positions = data["masked_lm_positions"]


    n_inst, seq_len = input_ids.shape
    n_inst2, def_len = d_input_ids.shape

    assert n_inst == n_inst2

    def_len = 128
    hidden_dim = 768
    reshaped_grad = reshape_gradienet(gradients, n_inst, def_len, hidden_dim)
    print(reshaped_grad.shape)

    n_pred = reshaped_grad.shape[1]

    grad_per_token = np.sum(np.abs(reshaped_grad), axis=3)

    p = os.path.join(output_path, "visualize", "dict_grad.html")
    f_html = open(p, "w", encoding="utf-8")
    f_html.write("<html><head>\n</head>\n")
    f_html.write("<body>\n")

    for inst_idx in range(n_inst):
        tokens = tokenizer.convert_ids_to_tokens(mask_input_ids[inst_idx])
        ans_tokens = tokenizer.convert_ids_to_tokens(input_ids[inst_idx])
        for i in range(len(tokens)):
            if tokens[i] == "[MASK]":
                tokens[i] = "[MASK_{}: {}]".format(i, ans_tokens[i])
            if tokens[i] == "[SEP]":
                tokens[i] = "[SEP]<br>"
        def_tokens = tokenizer.convert_ids_to_tokens(d_input_ids[inst_idx])
        s = tokenizer_wo_tf.pretty_tokens(tokens)

        lines = []

        grad_total_max = 0
        for pred_idx in range(n_pred):
            max_val = max(grad_per_token[inst_idx, pred_idx])
            total = sum(grad_per_token[inst_idx, pred_idx])
            mask_pos = masked_lm_positions[inst_idx, pred_idx]

            if total > grad_total_max:
                grad_total_max = total

            html_str = ""
            html_str += "<tr><td>{}</td>\n".format(mask_pos)
            html_str += "<td>{}</td>".format(int(total))

            for def_idx in range(def_len):
                term = def_tokens[def_idx]
                cont_right = def_idx +1 < def_len and def_tokens[def_idx][:2] == "##"
                cont_left = term[:2] == "##"

                space_left = "&nbsp;" if not cont_left else ""
                space_right = "&nbsp;" if not cont_right else ""

                if term == "[PAD]":
                    break
                if term == "[unused5]":
                    term = "[\\n]"
                bg_color = get_color(grad_per_token[inst_idx, pred_idx, def_idx]/(hidden_dim*2))
                html = "<td bgcolor=\"#{}\">{}{}{}</td>".format(bg_color, space_left, term, space_right)
                html_str += html
                print("{}({})".format(term, grad_per_token[inst_idx, pred_idx, def_idx]), end=" ")

            html_str += "</tr>\n"
            lines.append((mask_pos, html_str))
            print("")
        lines.sort(key=lambda x:x[0])


        f_html.write("<p>\n")
        s = s.replace("[unused4]", "<b>DictTerm</b>")
        f_html.write(s)
        f_html.write("</p>\n")
        if grad_total_max > 5000000:
            f_html.write("<h4>HIGH Gradient</h4>\n")
        f_html.write("<table>")
        for pos, line in lines:
            f_html.write(line)

        f_html.write("</table>")
        print("----------")
    f_html.write("</body></html>\n")



if __name__ == '__main__':
    load_and_analyze_gradient()

