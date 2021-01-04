from typing import List, Dict

from evals.types import TrecRankedListEntry
from list_lib import lmap, dict_value_map
from trec.trec_parse import load_ranked_list_grouped


def get_candidate_ids_from_ranked_list_path(ranked_list_path) -> Dict[str, List[str]]:
    rlg = load_ranked_list_grouped(ranked_list_path)
    return get_candidate_ids_from_ranked_list(rlg)


def get_candidate_ids_from_ranked_list(ranked_list: Dict[str, List[TrecRankedListEntry]]) \
        -> Dict[str, List[str]]:

    def get_ids(l: List[TrecRankedListEntry]) -> List[str]:
        return lmap(TrecRankedListEntry.get_doc_id, l)

    return dict_value_map(get_ids, ranked_list)