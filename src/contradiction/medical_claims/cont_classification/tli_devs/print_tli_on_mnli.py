import itertools
import logging
from typing import Dict, Tuple

from trainer_v2.per_project.tli.tli_visualize import til_to_table

from trainer_v2.per_project.tli.token_level_inference import TokenLevelInference, nc_max_e_avg_reduce_then_softmax
import numpy as np

from trainer_v2.per_project.tli.enum_subseq import enum_subseq_136
from dataset_specific.mnli.mnli_reader import MNLIReader
from explain.pairing.run_visualizer.show_cls_probe import NLIVisualize
from trainer_v2.chair_logging import c_log
from trainer_v2.keras_server.name_short_cuts import get_pep_client
from visualize.html_visual import HtmlVisualizer


def main():
    c_log.setLevel(logging.DEBUG)
    nli_predict_fn = get_pep_client()
    tli_module = TokenLevelInference(nli_predict_fn, enum_subseq_136)
    nli_visualize = NLIVisualize()

    reader = MNLIReader()
    pair_itr_gen = lambda : itertools.islice(reader.get_train(), 30)
    tli_payload = []
    for pair in pair_itr_gen():
        tli_payload.append((pair.premise, pair.hypothesis))

    tli_output_list = tli_module.do_batch(tli_payload)
    tli_dict: Dict[Tuple[str, str], np.array] = dict(zip(tli_payload, tli_output_list))
    html = HtmlVisualizer("tli_demo.html")
    for pair in pair_itr_gen():
        prem = pair.premise
        hypo = pair.hypothesis
        tli: np.array = tli_dict[prem, hypo]
        table = til_to_table(hypo, tli)

        probs_from_tli = nc_max_e_avg_reduce_then_softmax(tli)
        pred_str = nli_visualize.make_prediction_summary_str(probs_from_tli)
        html.write_paragraph("Prem: " + pair.premise)
        html.write_paragraph("Hypo: " + pair.hypothesis)
        html.write_paragraph("Gold: " + pair.label)
        html.write_paragraph("TLI based prediction: {}".format(pred_str))
        html.write_table(table)
        html.write_bar()


if __name__ == "__main__":
    main()