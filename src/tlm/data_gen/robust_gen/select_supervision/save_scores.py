import sys
from typing import List, Dict, Tuple

from arg.qck.qk_summarize import get_score_from_logit
from cache import save_to_pickle
from tlm.estimator_output_reader import load_combine_info_jsons, join_prediction_with_info


def summarize_score(info: Dict,
                    prediction_file_path: str,
                    score_type) -> Dict[Tuple[str, str, int], float]:
    key_logit = "logits"
    data: List[Dict] = join_prediction_with_info(prediction_file_path, info, ["data_id", key_logit])

    score_d: Dict[Tuple[str, str, int], float] = {}
    for entry in data:
        score = get_score_from_logit(score_type, entry['logits'])
        key = entry['query_id'], entry['doc_id'], entry['passage_idx']
        score_d[key] = score
    return score_d




def main():
    info_path = sys.argv[1]
    pred_path = sys.argv[2]
    save_name = sys.argv[3]
    score_type = "softmax"
    info = load_combine_info_jsons(info_path, False, False)
    score_d = summarize_score(info, pred_path, score_type)

    save_to_pickle(score_d, save_name)


if __name__ == "__main__":
    main()


