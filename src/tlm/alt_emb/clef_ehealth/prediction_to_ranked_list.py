from scipy.special import softmax

from galagos.parse_base import write_ranked_list_from_d


def prediction_to_ranked_list(pred_data, info, out_path):
    grouped = {}
    for entry in pred_data:
        data_id = entry.get_vector("data_id")[0]
        logits = entry.get_vector("logits")
        probs = softmax(logits)
        score = probs[1]

        data_id2, q_id, doc_id = info[data_id]
        assert data_id == data_id2
        key = q_id, doc_id
        if key not in grouped:
            grouped[key] = []

        grouped[key].append(score)

    group_by_q_id = {}
    for key, scores in grouped.items():
        reduced_score = max(scores)
        q_id, doc_id = key
        if q_id not in group_by_q_id:
            group_by_q_id[q_id] = []

        group_by_q_id[q_id].append((doc_id, reduced_score))
    ranked_list_d = {}
    for q_id in group_by_q_id:
        raw_ranked_list = group_by_q_id[q_id]
        raw_ranked_list.sort(key=lambda x: x[1], reverse=True)

        ranked_list = []
        for idx, (doc_id, score) in enumerate(raw_ranked_list):
            rank = idx + 1
            entry = (doc_id, rank, score)
            ranked_list.append(entry)

        ranked_list_d[q_id] = ranked_list

    write_ranked_list_from_d(ranked_list_d, out_path)

