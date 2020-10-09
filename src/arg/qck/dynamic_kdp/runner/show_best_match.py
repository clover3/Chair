from typing import List, Dict

from arg.perspectives.load import get_claim_perspective_id_dict2
from arg.qck.decl import qck_convert_map, QCKOutEntry
from arg.qck.doc_value_calculator import logit_to_score_softmax
from arg.qck.prediction_reader import load_combine_info_jsons
from arg.util import load_run_config
from estimator_helper.output_reader import join_prediction_with_info
from list_lib import lmap
from tab_print import print_table


def main():
    config = load_run_config()
    info = load_combine_info_jsons(config['info_path'], qck_convert_map, False)
    label_d: Dict[int, List[int]] = get_claim_perspective_id_dict2()
    print("Info length:", len(info))
    predictions: List[Dict] = join_prediction_with_info(config['pred_path'], info)
    print("Prediction length:", len(predictions))
    out_entries: List[QCKOutEntry] = lmap(QCKOutEntry.from_dict, predictions)

    out_entries = out_entries[:10000]
    out_entries.sort(key=lambda x: logit_to_score_softmax(x.logits), reverse=True)

    def get_label(entry: QCKOutEntry):
        return int(entry.candidate.id) in label_d[int(entry.query.query_id)]

    rows = []
    for entry in out_entries[:100]:
        label = get_label(entry)
        score = logit_to_score_softmax(entry.logits)
        print_info(entry, rows, score, label)
    print_table(rows)


def print_info(entry, rows, score, label):
    text = " ".join(entry.kdp.tokens)
    rows.append([])
    rows.append(["query", entry.query.query_id, entry.query.text])
    rows.append(["candidate", entry.candidate.id, entry.candidate.text])
    rows.append(["kdp", entry.kdp.doc_id, entry.kdp.passage_idx])
    rows.append(["label", label])
    rows.append([score])
    rows.append([text])


if __name__ == "__main__":
    main()