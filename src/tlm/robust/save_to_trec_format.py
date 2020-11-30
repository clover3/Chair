from typing import List, Tuple

from cache import load_pickle_from
from evals.trec import TrecRankedListEntry
from exec_lib import run_func_with_config
from galagos.parse import write_ranked_list_from_s
from tlm.robust.eval_metrics import get_payload_info, generate_ranked_list


def save_to_trec_format(prediction_path, payload_type, data_id, num_candidate, run_name, save_path):
    payload_info = get_payload_info(payload_type, data_id)
    tf_prediction_data = load_pickle_from(prediction_path)
    all_ranked_list = generate_ranked_list(tf_prediction_data, payload_info, num_candidate)
    st = int(data_id)
    query_ids = [str(i) for i in range(st, st + 50)]
    all_entries: List[Tuple[str, List[TrecRankedListEntry]]] = []
    for query_id, ranked_list in zip(query_ids, all_ranked_list):
        rl = [TrecRankedListEntry(query_id, doc_id, rank, score, run_name) for doc_id, rank, score in ranked_list]
        all_entries.append((query_id, rl))

    write_ranked_list_from_s(dict(all_entries), save_path)


def main(config):
    prediction_path = config['pred_path']
    payload_type = config['payload_type']
    data_id = config['data_id']
    num_candidate = config['num_candidate']
    run_name = config['run_name']
    save_path = config['save_path']
    save_to_trec_format(prediction_path, payload_type, data_id, num_candidate, run_name, save_path)


if __name__ == "__main__":
    run_func_with_config(main)