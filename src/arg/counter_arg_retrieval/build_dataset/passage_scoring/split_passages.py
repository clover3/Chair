from typing import Tuple, Dict, List

from bert_api.swtt.segmentwise_tokenized_text import SWTTIndex, SegmentwiseTokenizedText
from bert_api.swtt.window_enum_policy import WindowEnumPolicyFixedLimit
from trec.types import DocID, TrecRankedListEntry

PassageRange = Tuple[SWTTIndex, SWTTIndex]


def split_passages(docs,
                   ranked_list_groups: Dict[DocID, List[TrecRankedListEntry]],
                   enum_policy: WindowEnumPolicyFixedLimit,
                   ) -> Dict[DocID, Tuple[SegmentwiseTokenizedText, List[PassageRange]]]:
    docs_d = dict(docs)
    n_not_found = 0
    doc_as_passage_d = {}
    for qid, ranked_list in ranked_list_groups.items():
        ranked_list = ranked_list_groups[qid]
        doc_id_list = list(map(TrecRankedListEntry.get_doc_id, ranked_list))
        for doc_id in doc_id_list:
            try:
                doc = docs_d[doc_id]
                window_list: List[Tuple[SWTTIndex, SWTTIndex]] = enum_policy.window_enum(doc)
                doc_as_passage = doc, window_list
                doc_as_passage_d[doc_id] = doc_as_passage
            except KeyError:
                n_not_found += 1
                print(f"Doc {doc_id} not found")
                if n_not_found > 10:
                    raise
    return doc_as_passage_d