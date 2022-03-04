from typing import List, Dict

from list_lib import dict_key_map
from trec.trec_parse import score_d_to_ranked_list_entries


def convert_token_scores_to_trec_entries(query_id, run_name, token_scores: List[float]):
    token_scores_d = {idx: s for idx, s in enumerate(token_scores)}
    token_scores_s: Dict[str, float] = dict_key_map(str, token_scores_d)
    ranked_list = score_d_to_ranked_list_entries(token_scores_s, run_name, query_id)
    return ranked_list


def convert_token_score_d_to_trec_entries(query_id, run_name, token_scores: Dict[int, float]):
    token_scores_s: Dict[str, float] = dict_key_map(str, token_scores)
    ranked_list = score_d_to_ranked_list_entries(token_scores_s, run_name, query_id)
    return ranked_list