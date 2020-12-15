from cache import load_pickle_from
from trec.trec_parse import TrecRankedListEntry
from galagos.parse import write_ranked_list_from_s
from tlm.robust.eval_metrics import get_payload_info, generate_ranked_list


def save_to_trec_format():
    prediction_path_format = "output/robust/A_train_{}.score"
    payload_type = "first_clean"
    num_candidate = 100
    run_name = "train_eval"
    save_path = "output/ranked_list/robust_train_pred.txt"
    all_entries = []
    for data_id in [301, 351, 401, 601]:
        payload_info = get_payload_info(payload_type, str(data_id))
        prediction_path = prediction_path_format.format(data_id)
        tf_prediction_data = load_pickle_from(prediction_path)
        all_ranked_list = generate_ranked_list(tf_prediction_data, payload_info, num_candidate)
        st = int(data_id)
        query_ids = [str(i) for i in range(st, st + 50)]
        for query_id, ranked_list in zip(query_ids, all_ranked_list):
            rl = [TrecRankedListEntry(query_id, doc_id, rank, score, run_name) for doc_id, rank, score in ranked_list]
            all_entries.append((query_id, rl))

    write_ranked_list_from_s(dict(all_entries), save_path)


if __name__ == "__main__":
    save_to_trec_format()