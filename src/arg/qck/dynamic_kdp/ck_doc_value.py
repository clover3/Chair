from typing import List, Dict

import scipy.special

from arg.qck.decl import get_format_handler, QCKOutEntry
from arg.qck.prediction_reader import load_combine_info_jsons
from exec_lib import run_func_with_config
from list_lib import lmap
from misc_lib import group_by, unique_list
from tab_print import print_table
from tlm.estimator_output_reader import join_prediction_with_info
from trec.qrel_parse import load_qrels_structured


def main(config):
    info_dir = config['info_path']
    prediction_file = config['pred_path']

    f_handler = get_format_handler("qck")
    info = load_combine_info_jsons(info_dir, f_handler.get_mapping(), f_handler.drop_kdp())
    data: List[Dict] = join_prediction_with_info(prediction_file, info, ["data_id", "logits"])
    out_entries: List[QCKOutEntry] = lmap(QCKOutEntry.from_dict, data)
    qrel: Dict[str, Dict[str, int]] = load_qrels_structured(config['qrel_path'])

    def get_label(query_id, candi_id):
        if candi_id in qrel[query_id]:
            return qrel[query_id][candi_id]
        else:
            return 0

    def logit_to_score_softmax(logit):
        return scipy.special.softmax(logit)[1]

    grouped: Dict[str, List[QCKOutEntry]] = group_by(out_entries, lambda x: x.query.query_id)
    for query_id, items in grouped.items():
        raw_kdp_list = [(x.kdp.doc_id, x.kdp.passage_idx) for x in items]
        kdp_list = unique_list(raw_kdp_list)

        raw_candi_id_list = [x.candidate.id for x in items]
        candi_id_list = unique_list(raw_candi_id_list)

        logit_d = {(x.candidate.id, (x.kdp.doc_id, x.kdp.passage_idx)): x.logits for x in items}
        labels = [get_label(query_id, candi_id) for candi_id in candi_id_list]
        head_row0 = [" "] + labels
        head_row1 = [" "] + candi_id_list
        rows = [head_row0, head_row1]
        for kdp_sig in kdp_list:
            row = [kdp_sig]
            for candi_id in candi_id_list:
                try:
                    score = logit_to_score_softmax(logit_d[candi_id, kdp_sig])
                    score_str = "{0:.2f}".format(score)
                except KeyError:
                    score_str = "-"
                row.append(score_str)
            rows.append(row)

        print(query_id)
        print_table(rows)


if __name__ == "__main__":
    run_func_with_config(main)