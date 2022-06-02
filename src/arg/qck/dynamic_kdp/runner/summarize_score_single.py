import pickle
import sys
from typing import List, Dict, Tuple

from arg.perspectives.doc_value_viewer.calculate_doc_score import load_labels
from arg.qck.decl import qck_convert_map, QCKOutEntry
from arg.qck.doc_value_calculator import get_doc_value_parts2, \
    DocValueParts2
from arg.qck.dynamic_kdp.score_summarizer import load_baseline
from arg.qck.prediction_reader import load_combine_info_jsons
from list_lib import lmap
from tlm.estimator_output_reader import join_prediction_with_info


def main():
    baseline_score: Dict[Tuple[str, str], float] = load_baseline()
    score_save_path = sys.argv[1]
    info_path = sys.argv[2]
    info = load_combine_info_jsons(info_path, qck_convert_map, False)
    # calculate score for each kdp
    predictions: List[Dict] = join_prediction_with_info(score_save_path, info, ["logits"], True)
    out_entries: List[QCKOutEntry] = lmap(QCKOutEntry.from_dict, predictions)
    labels: Dict[str, List[str]] = load_labels()
    doc_score_parts: List[DocValueParts2] = get_doc_value_parts2(out_entries, baseline_score, labels)
    summary_save_path = sys.argv[3]
    pickle.dump(doc_score_parts, open(summary_save_path, "wb"))


if __name__ == "__main__":
    main()