from typing import List, Dict, NamedTuple

from exec_lib import run_func_with_config
from list_lib import lmap
from misc_lib import group_by
from tab_print import tab_print
from tlm.estimator_output_reader import load_combine_info_jsons, join_prediction_with_info


class OutEntry(NamedTuple):
    text: str
    logits : List
    doc_id: str
    sent_idx: int
    @classmethod
    def from_dict(cls, d):
        return OutEntry(d['sentence'], d['logits'], d['doc_id'], d['sent_idx'])


def main(config):
    info = load_combine_info_jsons(config['info_path'])
    predictions: List[Dict] = join_prediction_with_info(config['pred_path'], info)
    entries = lmap(OutEntry.from_dict, predictions)

    def get_doc_id(e: OutEntry):
        return e.doc_id

    grouped = group_by(entries, get_doc_id)

    for doc_id in grouped:
        doc_entries = grouped[doc_id]
        doc_entries.sort(key=lambda x: x.sent_idx)
        n_pos = 0
        for s in doc_entries:
            if s.logits[1] > 0.5:
                n_pos += 1
        tab_print(doc_id, n_pos, len(doc_entries))


if __name__ == "__main__":
    run_func_with_config(main)