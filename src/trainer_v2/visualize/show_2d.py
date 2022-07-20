import sys
from typing import List

import numpy as np

from cache import load_from_pickle
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem, AlamriProblem
from data_generator.tokenize_helper import TokenizedText
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lmap
from trainer_v2.custom_loop.per_task.nli_ms_util import get_local_decision_nlims
from trainer_v2.custom_loop.run_config2 import get_run_config2
from trainer_v2.train_util.arg_flags import flags_parser
from visualize.html_visual import HtmlVisualizer, Cell
from visualize.nli_visualize import get_cell_str, prob_to_color


def main(args):
    run_config = get_run_config2(args)
    nlims = get_local_decision_nlims(run_config)
    problems: List[AlamriProblem] = load_alamri_problem()
    tokenizer = get_tokenizer()

    html = HtmlVisualizer("nlims.html")
    for p in problems:
        t1 = p.text1.split()
        t2 = p.text2.split()
        tt1 = TokenizedText.from_word_tokens(" ".join(t1), tokenizer, t1)
        tt2 = TokenizedText.from_word_tokens(" ".join(t2), tokenizer, t2)
        l1 = len(tt1.sbword_tokens) + 2
        l2 = len(tt2.sbword_tokens) + 2
        # x = nlims.encode_fn(tt1.sbword_tokens, tt2.sbword_tokens)
        # es = TokenizedTextBasedES(tt1, tt2, x)
        # output_list = nlims.predict([x])
        # neural_output = output_list[0]
        def get_token(i, tokens):
            if i == 0:
                return "[CLS]"
            elif i == len(tokens) + 1:
                return "[SEP]"
            else:
                return tokens[i-1]

        local_decision = load_from_pickle("local_decision")[0]
        head = lmap(Cell, ["", "[CLS]"] + tt2.sbword_tokens + ["[SEP]"])
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
            table.append(row)

        table.append([Cell("---")])
        max_row = [Cell("Max: ")]
        for i2 in range(l2):
            item = np.max(local_decision[:, i2], axis=0)

            max_row.append(Cell(get_cell_str(item)))
        table.append(max_row)
        html.write_table(table)
        print(local_decision)
        # save_to_pickle(local_decision, "local_decision")

        break


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
