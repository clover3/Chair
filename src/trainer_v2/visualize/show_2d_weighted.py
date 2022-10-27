import sys
from typing import List

import numpy as np
import scipy
import scipy.special

from cache import load_cache, save_to_pickle
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem, AlamriProblem
from data_generator.tokenize_helper import TokenizedText
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import two_digit_float
from trainer_v2.custom_loop.per_task.nli_ms_util import get_weighted_nlims
from trainer_v2.custom_loop.run_config2 import get_eval_run_config2
from trainer_v2.train_util.arg_flags import flags_parser
from visualize.html_visual import HtmlVisualizer, Cell
from visualize.nli_visualize import get_cell_str, prob_to_color


def main(args):
    run_config = get_eval_run_config2(args)
    nlims = None
    problems: List[AlamriProblem] = load_alamri_problem()
    tokenizer = get_tokenizer()

    html = HtmlVisualizer("{}.html".format(run_config.common_run_config.run_name))
    for p in problems:
        t1 = p.text1.split()
        t2 = p.text2.split()
        tt1 = TokenizedText.from_word_tokens(" ".join(t1), tokenizer, t1)
        tt2 = TokenizedText.from_word_tokens(" ".join(t2), tokenizer, t2)
        # x = nlims.encode_fn(tt1.sbword_tokens, tt2.sbword_tokens)
        # es = TokenizedTextBasedES(tt1, tt2, x)
        data_name = "visual_temp_{}_{}".format(run_config.common_run_config.run_name, p.data_id)
        outputs = load_cache(data_name)
        if outputs is None:
            if nlims is None:
                nlims = get_weighted_nlims(run_config)
            x = nlims.encode_fn(tt1.sbword_tokens, tt2.sbword_tokens)
            outputs = nlims.predict([x])
            save_to_pickle(outputs, data_name)

        local_decision_b, weights_b, g_decision_b = outputs
        local_decision = local_decision_b[0]
        weights = scipy.special.softmax(weights_b[0], axis=0)
        print(weights.shape)

        l1 = len(tt1.sbword_tokens) + 2
        l2 = len(tt2.sbword_tokens) + 2
        def get_token(i, tokens):
            if i == 0:
                return "[CLS]"
            elif i == len(tokens) + 1:
                return "[SEP]"
            else:
                return tokens[i-1]
        head = [Cell(""), Cell("[CLS]"), Cell("")]
        for t in tt2.sbword_tokens:
            head.append(Cell(t))
            head.append(Cell(""))
        head.append(Cell("[SEP]"))
        table = [head]
        for i1 in range(l1):
            token1 = get_token(i1, tt1.sbword_tokens)
            row = [Cell(token1)]
            for i2 in range(l2):
                item = local_decision[i1, i2]
                color_score = prob_to_color(item)
                color = "".join([("%02x" % int(v)) for v in color_score])
                c = Cell(get_cell_str(item), 255, target_color=color)
                row.append(c)
                w = weights[i1, i2][0]
                c = Cell(two_digit_float(w), int(w * 100))
                row.append(c)
            table.append(row)

        table.append([Cell("---")])
        wsum = [Cell("WSum: ")]
        for i2 in range(l2):
            token2 = get_token(i2, tt2.sbword_tokens)
            max_i1 = np.argmax(weights[:, i2])
            if max_i1 >= len(tt1.sbword_tokens):
                print("Max for {}-th token {} at {}".format(i2, token2, max_i1))
                max_t1 = "PAD"
            else:
                max_t1 = get_token(max_i1, tt1.sbword_tokens)
            item = np.sum(local_decision[:, i2] * weights[:, i2], axis=0)
            color_score = prob_to_color(item)
            color = "".join([("%02x" % int(v)) for v in color_score])
            c = Cell(get_cell_str(item), 255, target_color=color)
            wsum.append(c)
            wsum.append(Cell(max_t1))
        table.append(wsum)
        html.write_table(table)

        break

if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
