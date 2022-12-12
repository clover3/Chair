import itertools
import logging
from typing import List
from typing import List, Iterable, Callable, Dict, Tuple, Set

from contradiction.medical_claims.cont_classification.solvers.token_nli import TokenLevelInference, \
    max_reduce_then_softmax, nc_max_e_avg_reduce_then_softmax
import numpy as np
from contradiction.medical_claims.retrieval.nli_system import enum_subseq_136
from dataset_specific.mnli.mnli_reader import MNLIReader
from explain.pairing.run_visualizer.show_cls_probe import NLIVisualize
from trainer_v2.chair_logging import c_log
from trainer_v2.keras_server.name_short_cuts import get_pep_client
from visualize.html_visual import HtmlVisualizer, Cell


def main():
    c_log.setLevel(logging.DEBUG)
    nli_predict_fn = get_pep_client()
    tli_module = TokenLevelInference(nli_predict_fn, enum_subseq_136)
    nli_visualize = NLIVisualize()
    def prob_to_color(prob) -> str:
        color_score = nli_visualize.prob_to_color(prob)
        color = "".join([("%02x" % int(v)) for v in color_score])
        return color

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
        color_array: List[str] = list(map(prob_to_color, tli))
        cell_str_array = list(map(nli_visualize.get_cell_str, tli))
        row1 = [Cell(t) for t in hypo.split()]
        row2 = []
        row3 = []
        for cell_str, color in zip(cell_str_array, color_array):
            cell = Cell(cell_str, 255, target_color=color)
            row2.append(cell)
            row3.append(Cell(cell_str))

        probs_from_tli = nc_max_e_avg_reduce_then_softmax(tli)
        pred_str = nli_visualize.make_prediction_summary_str(probs_from_tli)
        table = [row1, row2, row3]
        html.write_paragraph("Prem: " + pair.premise)
        html.write_paragraph("Hypo: " + pair.hypothesis)
        html.write_paragraph("Gold: " + pair.label)
        html.write_paragraph("TLI based prediction: {}".format(pred_str))
        html.write_table(table)
        html.write_bar()


if __name__ == "__main__":
    main()