import os
from typing import List, Dict, Tuple

# Input: Ranked List, Enum Policy, SWTT
# Output:   Dict[doc_id, [SWTT, List[SWTTScorerInput]]]
from bert_api.swtt.segmentwise_tokenized_text import SWTTIndex, SegmentwiseTokenizedText
from bert_api.swtt.window_enum_policy import get_run3_enum_policy, WindowEnumPolicyFixedLimit
from cache import load_from_pickle, save_to_pickle
from cpath import output_path
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry, DocID

PassageRange = Tuple[SWTTIndex, SWTTIndex]


def split_passages(docs,
                   ranked_list_groups: Dict[DocID, List[TrecRankedListEntry]],
                   enum_policy: WindowEnumPolicyFixedLimit,
                   ) -> Dict[DocID, Tuple[SegmentwiseTokenizedText, List[PassageRange]]]:
    docs_d = dict(docs)
    doc_as_passage_d = {}
    for qid, ranked_list in ranked_list_groups.items():
        ranked_list = ranked_list_groups[qid]
        doc_id_list = list(map(TrecRankedListEntry.get_doc_id, ranked_list))
        for doc_id in doc_id_list:
            doc = docs_d[doc_id]
            window_list: List[Tuple[SWTTIndex, SWTTIndex]] = enum_policy.window_enum(doc)
            doc_as_passage = doc, window_list
            doc_as_passage_d[doc_id] = doc_as_passage
    return doc_as_passage_d


def load_ca3_swtt_passage():
    return load_from_pickle("ca_run3_swtt_passages")


def main():
    enum_policy = get_run3_enum_policy()
    docs: List[Tuple[str, SegmentwiseTokenizedText]] = load_from_pickle("ca_run3_swtt")
    rlg_path = os.path.join(output_path, "ca_building", "run3", "q_res_2.filtered.txt")
    rlg = load_ranked_list_grouped(rlg_path)
    doc_as_passage_dict = split_passages(docs, rlg, enum_policy)
    save_to_pickle(doc_as_passage_dict, "ca_run3_swtt_passages")


if __name__ == "__main__":
    main()
