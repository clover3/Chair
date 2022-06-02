from typing import List, Tuple

from scipy.special import softmax

from cache import load_from_pickle
from cpath import pjoin, output_path
from galagos.parse_base import write_ranked_list_from_d
from list_lib import dict_value_map
from tlm.estimator_prediction_viewer import EstimatorPredictionViewer


def save_ranked_list(prediction_path, meta_info, save_path):

    data = EstimatorPredictionViewer(prediction_path)

    q_dict = {}
    for entry in data:
        data_id = entry.get_vector('data_id')[0]
        scores = entry.get_vector('logits')
        q_id, doc_id = meta_info[data_id]

        if q_id not in q_dict:
            q_dict[q_id] = []

        probs = softmax(scores)
        q_dict[q_id].append((doc_id, probs[1]))

    def add_rank(ranked_list: List[Tuple[str, float]]) -> List[Tuple[str, int, float]]:
        ranked_list.sort(key=lambda x:x[1], reverse=True)
        ranked_list = [(doc_id, rank, score) for rank, (doc_id, score) in enumerate(ranked_list)]
        return ranked_list

    q_dict_new = dict_value_map(add_rank, q_dict)
    write_ranked_list_from_d(q_dict_new, save_path)


def main():
    prediction_path = pjoin(output_path, "bert_msmarco_cord_2_4")
    meta_info = load_from_pickle("data_info_save")
    save_path = pjoin(output_path, "bert_msmarco_cord_2_4.txt")
    save_ranked_list(prediction_path, meta_info, save_path)


if __name__ == "__main__":
    main()