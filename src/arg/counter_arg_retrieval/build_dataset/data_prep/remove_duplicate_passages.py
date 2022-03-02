from typing import Dict, Tuple, List

from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText, SWTTIndex
from list_lib import flatten
from misc_lib import get_duplicate_list
from trec.trec_parse import scores_to_ranked_list_entries
from trec.types import TrecRankedListEntry

SplitDocDict = Dict[str, Tuple[SegmentwiseTokenizedText, List[Tuple[SWTTIndex, SWTTIndex]]]]


def remove_duplicate_passages(ranked_list_grouped: Dict[str, List[TrecRankedListEntry]],
                              docs_d: SplitDocDict) -> Dict[str, List[TrecRankedListEntry]]:
    def get_rep(passage_doc_id) -> str:
        tokens = passage_doc_id.split("_")
        doc_id = "_".join(tokens[:-1])
        passage_idx = int(tokens[-1])
        swtt, passage_ranges = docs_d[doc_id]
        st, ed = passage_ranges[passage_idx]
        tokens_list: List[List[str]] = swtt.get_word_tokens_grouped(st, ed)
        return " ".join(flatten(tokens_list))

    new_rlg = {}
    for qid, entries in ranked_list_grouped.items():
        passage_doc_ids: List[str] = [e.doc_id for e in entries]
        duplicate_indices: List[int] = get_duplicate_list(map(get_rep, passage_doc_ids))
        unique_entries = [e for idx, e in enumerate(entries) if idx not in duplicate_indices]
        unique_entries_raw: List[Tuple[str, float]] = [(e.doc_id, e.score) for e in unique_entries]
        run_name = entries[0].run_name
        new_entries = scores_to_ranked_list_entries(unique_entries_raw, run_name, qid)
        new_rlg[qid] = new_entries
    return new_rlg