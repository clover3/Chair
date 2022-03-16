from list_lib import dict_key_map, left
from typing import List, Dict, Tuple

from list_lib import dict_key_map, left
from trec.qrel_parse import load_qrels_flat_per_query
from trec.trec_parse import score_d_to_ranked_list_entries
from trec.types import QueryID, DocID


def convert_token_scores_to_trec_entries(query_id, run_name, token_scores: List[float]):
    token_scores_d = {idx: s for idx, s in enumerate(token_scores)}
    token_scores_s: Dict[str, float] = dict_key_map(str, token_scores_d)
    ranked_list = score_d_to_ranked_list_entries(token_scores_s, run_name, query_id)
    return ranked_list


def convert_token_score_d_to_trec_entries(query_id, run_name, token_scores: Dict[int, float]):
    token_scores_s: Dict[str, float] = dict_key_map(str, token_scores)
    ranked_list = score_d_to_ranked_list_entries(token_scores_s, run_name, query_id)
    return ranked_list


def convert_trec_labels_to_list(label_path):
    qrels: Dict[QueryID, List[Tuple[DocID, int]]] = load_qrels_flat_per_query(label_path)

    for qid, entries in qrels.items():
        token_labels = [(int(doc_id_s), label) for doc_id_s, label in entries]
        max_idx = max(left(token_labels))
        seq_len = max_idx + 1







