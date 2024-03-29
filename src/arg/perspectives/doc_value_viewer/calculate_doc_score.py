from typing import List, Dict, Tuple

from arg.perspectives.load import get_claim_perspective_id_dict2
from arg.qck.decl import QCKOutEntry
from arg.qck.doc_value_calculator import get_doc_value_parts, DocValueParts
from list_lib import lmap
from tlm.estimator_output_reader import join_prediction_with_info


def load_labels() -> Dict[str, List[str]]:
    return {str(k): lmap(str, v) for k, v in get_claim_perspective_id_dict2().items()}


def calculate_score(info,
                    pred_path,
                    baseline_score: Dict[Tuple[str, str], float],
                    str_data_id = False
                    ) -> List[DocValueParts]:

    predictions: List[Dict] = join_prediction_with_info(pred_path, info, ["logits"], str_data_id)
    out_entries: List[QCKOutEntry] = lmap(QCKOutEntry.from_dict, predictions)
    labels: Dict[str, List[str]] = load_labels()
    doc_score_parts: List[DocValueParts] = get_doc_value_parts(out_entries, baseline_score, labels)
    return doc_score_parts
