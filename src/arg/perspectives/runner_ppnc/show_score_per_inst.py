import os
import sys
from typing import List, Dict, Tuple

from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import get_claims_from_ids, load_dev_claim_ids, get_claim_perspective_id_dict
from arg.perspectives.ppnc.get_doc_value import group_by_cpid
from arg.perspectives.types import CPIDPair
from cpath import output_path
from data_generator.tokenizer_wo_tf import pretty_tokens
from list_lib import lmap
from misc_lib import group_by, exist_or_mkdir, average
from visualize.html_visual import Cell, HtmlVisualizer, get_tooltip_cell


def main():
    save_name = sys.argv[1]
    out_dir = os.path.join(output_path, "cppnc")
    exist_or_mkdir(out_dir)
    info_file_path = os.path.join(out_dir, save_name + ".info")
    pred_file_path = os.path.join(out_dir, save_name + ".score")
    scores: Dict[CPIDPair, List[Dict]] = group_by_cpid(info_file_path, pred_file_path)

    d_ids = list(load_dev_claim_ids())
    claims: List[Dict] = get_claims_from_ids(d_ids)
    claim_d = {c['cId']: c['text'] for c in claims}

    html = HtmlVisualizer("detail_view.html", dark_mode=False, use_tooltip=True)
    cid_grouped: Dict[int, List[Tuple[CPIDPair, List[Dict]]]] = group_by(scores.items(), lambda x: x[0][0])
    gold = get_claim_perspective_id_dict()

    for cid, pid_entries in cid_grouped.items():
        gold_pids = gold[cid]

        def get_score_per_pid_entry(p_entries: Tuple[CPIDPair, List[Dict]]):
            cpid, entries = p_entries
            return average(lmap(lambda e: e['score'], entries))

        pid_entries.sort(key=get_score_per_pid_entry, reverse=True)

        s = "{} : {}".format(cid, claim_d[cid])
        html.write_headline(s)
        rows = []
        for cpid, things in pid_entries:
            _, pid = cpid
            correct = any([pid in pids for pids in gold_pids])
            head = [Cell(pid),
                    Cell("Y" if correct else "N"),
                    Cell(perspective_getter(pid)),
                    ]
            scores_str: List[str] = lmap(lambda x: "{0:.4f}".format(x['score']), things)
            passages: List[str] = lmap(lambda x: pretty_tokens(x['passage'], drop_sharp=True), things)

            tail = lmap(lambda x: get_tooltip_cell(x[0], x[1]), zip(scores_str, passages))
            row = head + tail
            rows.append(row)
        html.write_table(rows)


if __name__ == "__main__":
    main()

