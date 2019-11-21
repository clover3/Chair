import pickle
import numpy as np
from path import output_path, data_path
import os
from data_generator import tokenizer_wo_tf
from misc_lib import right


def load_and_analyze_gradient():
    p = os.path.join(output_path, "dict_grad2.pickle")
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


class Cell:
    def __init__(self, s, highlight_score=0, space_left=True, space_right=True):
        self.s = str(s)

        # score should be normalized to 0~255 scale, or else floor
        if highlight_score > 255:
            highlight_score = 255
        elif highlight_score < 0:
            highlight_score = 0
        self.highlight_score = highlight_score
        self.space_left = space_left
        self.space_right = space_right


class HtmlVisualizer:
    def __init__(self, filename, dark_mode=False):
        p = os.path.join(output_path, "visualize", filename)
        self.f_html = open(p, "w", encoding="utf-8")
        self.dark_mode = dark_mode
        self.dark_foreground = "A9B7C6"
        self.dark_background = "2B2B2B"
        self._write_header()

    def _write_header(self):
        self.f_html.write("<html><head>\n")

        if self.dark_mode:
            self.f_html.write("<style>body{color:#" + self.dark_foreground + ";}</style>")

        self.f_html.write("</head>\n")

        if self.dark_mode:
            self.f_html.write("<body style=\"background-color:#{};\"".format(self.dark_background))
        else:
            self.f_html.write("<body>\n")

    def close(self):
        self.f_html.write("</body>\n")
        self.f_html.write("</html>\n")

    def write_paragraph(self, s):
        self.f_html.write("<p>\n")
        self.f_html.write(s+"\n")
        self.f_html.write("</p>\n")

    def write_headline(self, s, level=4):
        self.f_html.write("<h{}>{}</h{}>\n".format(level, s, level))

    def write_table(self, rows):
        self.f_html.write("<table>\n")

        for row in rows:
            self.f_html.write("<tr>\n")
            for cell in row:
                s = self.get_cell_html(cell)
                self.f_html.write(s)
            self.f_html.write("</tr>\n")
        self.f_html.write("</table>\n")

    def get_cell_html(self, cell):
        left = "&nbsp;" if cell.space_left else ""
        right = "&nbsp;" if cell.space_right else ""
        if cell.highlight_score:
            if not self.dark_mode:
                bg_color = self.get_blue(cell.highlight_score)
            else:
                bg_color = self.get_blue_d(cell.highlight_score)

            s = "<td bgcolor=\"#{}\">{}{}{}</td>".format(bg_color, left, cell.s, right)
        else:
            s = "<td>{}{}{}</td>".format(left, cell.s, right)

        return s

    def get_blue(self, r):
        r = 255 - int(r)
        bg_color = ("%02x" % r) + ("%02x" % r) + "ff"
        return bg_color

    def get_blue_d(self, r):
        r = (0xFF - 0x2B) * r / 255
        r = 0x2B + int(r)
        bg_color = "2B2B" + ("%02x" % r)
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

    def_len = 256
    hidden_dim = 768
    reshaped_grad = reshape_gradienet(gradients, n_inst, def_len, hidden_dim)
    print(reshaped_grad.shape)

    n_pred = reshaped_grad.shape[1]

    grad_per_token = np.sum(np.abs(reshaped_grad), axis=3)

    html_writer = HtmlVisualizer("dict_grad.html", dark_mode=False)


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
            row = []
            max_val = max(grad_per_token[inst_idx, pred_idx])
            total = sum(grad_per_token[inst_idx, pred_idx])
            mask_pos = masked_lm_positions[inst_idx, pred_idx]

            if total > grad_total_max:
                grad_total_max = total


            row.append(Cell(mask_pos))
            row.append(Cell(int(total)))

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

                score = grad_per_token[inst_idx, pred_idx, def_idx]/(hidden_dim*2)
                bg_color = get_color(score)

                row.append(Cell(term, score, not cont_left, not cont_right))
                print("{}({})".format(term, grad_per_token[inst_idx, pred_idx, def_idx]), end=" ")

            lines.append((mask_pos, row))
            print("")
        lines.sort(key=lambda x:x[0])

        s = s.replace("[unused4]", "<b>DictTerm</b>")
        html_writer.write_paragraph(s)

        if grad_total_max > 5000000:
            html_writer.write_headline("HIGH Gradient")

        rows = right(lines)
        html_writer.write_table(rows)

        print("----------")
    html_writer.close()


if __name__ == '__main__':
    load_and_analyze_gradient()

