import sys
from typing import List
from typing import NamedTuple

from estimator_helper.output_reader import load_combine_info_jsons, join_prediction_with_info
from exec_lib import run_func_with_config
from list_lib import lmap
from misc_lib import group_by
from scipy_aux import logit_to_score_softmax


class OutEntry(NamedTuple):
    doc_id: str
    passage_idx: int
    logits: List
    query_id: str

    @classmethod
    def from_dict(cls, d):
        return OutEntry(d['doc_id'], d['passage_idx'], d['logits'], d['query_id'])


def main(config):
    info_path = sys.argv[1]
    pred_path = sys.argv[2]

    info = load_combine_info_jsons(info_path, True)
    predictions = join_prediction_with_info(pred_path, info, silent=True)
    out_entries: List[OutEntry] = lmap(OutEntry.from_dict, predictions)
    g = group_by(out_entries, lambda x: x.doc_id)

    for doc_id in g:
        entries: List[OutEntry] = g[doc_id]
        scores = list([logit_to_score_softmax(e.logits) for e in entries])
        print(doc_id, max(scores))



if __name__ == "__main__":
    run_func_with_config(main)