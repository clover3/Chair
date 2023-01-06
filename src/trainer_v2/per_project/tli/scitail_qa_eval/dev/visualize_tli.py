from typing import List, Iterable, Callable, Dict, Tuple, Set
from dataset_specific.scitail import ScitailEntry, load_scitail_structured
from misc_lib import average
from trainer_v2.keras_server.name_short_cuts import get_pep_cache_client
from trainer_v2.per_project.tli.enum_subseq import enum_subseq_136
from trainer_v2.per_project.tli.tli_visualize import til_to_table
from trainer_v2.per_project.tli.token_level_inference import TokenLevelInference, Numpy2D, \
    nc_max_e_avg_reduce_then_softmax, nc_max_e_avg_reduce_then_norm

from visualize.html_visual import Cell, HtmlVisualizer


def main():
    split = "train"
    entries: List[ScitailEntry] = load_scitail_structured(split)
    entries = entries[:30]
    nli_predict_fn = get_pep_cache_client()
    tli_module = TokenLevelInference(nli_predict_fn, enum_subseq_136)

    payload = [(e.sentence1, e.question) for e in entries]
    tli_d: Dict[Tuple[str, str], Numpy2D] = tli_module.do_batch_return_dict(payload)

    html = HtmlVisualizer("scitail_tnli.html")
    for e in entries:
        tli: Numpy2D = tli_d[e.sentence1, e.question]
        table: List[List[Cell]] = til_to_table(e.question, tli)

        e_sum = sum(tli[:, 0])
        e_avg = average(tli[:, 0])
        probs = nc_max_e_avg_reduce_then_norm(tli)

        html.write_paragraph("Sentence 1: " + e.sentence1)
        html.write_paragraph("question: " + e.question)
        html.write_paragraph("Gold : " + e.label)
        html.write_paragraph("sum={0:2f}, avg={1:.2f}, probs={2}".format(e_sum, e_avg, probs))
        html.write_table(table)
        html.write_bar()


if __name__ == "__main__":
    main()