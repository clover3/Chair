from typing import List, Dict, NamedTuple

from exec_lib import run_func_with_config
from list_lib import lmap
from scipy_aux import logit_to_score_softmax
from tab_print import print_table
from tlm.estimator_output_reader import load_combine_info_jsons, join_prediction_with_info


class OutEntry(NamedTuple):
    cid: int
    pid: int
    logits: List
    doc_id: str
    sent_idx: int
    @classmethod
    def from_dict(cls, d):
        return OutEntry(d['cid'], d['pid'], d['logits'], d['doc_id'], d['sent_idx'])


def main(config):
    info = load_combine_info_jsons(config['info_path'])
    predictions: List[Dict] = join_prediction_with_info(config['pred_path'], info,
                                                        ["data_ids", "logits"], True, "data_ids")
    entries: List[OutEntry] = lmap(OutEntry.from_dict, predictions)

    def is_pos(e: OutEntry):
        return logit_to_score_softmax(e.logits) > 0.5
    pos_entries = filter(is_pos, entries)

    rows = []
    for e in pos_entries:
        row = [e.cid, e.pid, e.doc_id, e.sent_idx]
        rows.append(row)
    print_table(rows)


if __name__ == "__main__":
    run_func_with_config(main)