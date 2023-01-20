import itertools
import logging
from typing import Dict, Tuple
from trainer_v2.per_project.tli.tli_visualize import til_to_table

from trainer_v2.per_project.tli.token_level_inference import TokenLevelInference, nc_max_e_avg_reduce_then_softmax
import numpy as np
from trainer_v2.per_project.tli.enum_subseq import enum_subseq_136
from explain.pairing.run_visualizer.show_cls_probe import NLIVisualize
from trainer_v2.chair_logging import c_log
from trainer_v2.keras_server.name_short_cuts import get_pep_client
from visualize.html_visual import HtmlVisualizer


def main():
    c_log.setLevel(logging.DEBUG)
    nli_predict_fn = get_pep_client()
    tli_module = TokenLevelInference(nli_predict_fn, enum_subseq_136)
    nli_visualize = NLIVisualize()
    html = HtmlVisualizer("tli_console.html")
    while True:
        p = input("Premise: ")
        h = input("Hypothesis: ")

        outputs = tli_module.do_batch([(p, h)])
        tli = outputs[0]
        table = til_to_table(h, tli)

        probs_from_tli = nc_max_e_avg_reduce_then_softmax(tli)
        pred_str = nli_visualize.make_prediction_summary_str(probs_from_tli)
        html.write_paragraph("Prem: " + p)
        html.write_paragraph("Hypo: " + h)
        html.write_paragraph("TLI based prediction: {}".format(pred_str))
        html.write_table(table)
        html.write_bar()
        html.f_html.flush()


if __name__ == "__main__":
    main()