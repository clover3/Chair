import os
from typing import List, Tuple

# Input: Ranked List, Enum Policy, SWTT
# Output:   Dict[doc_id, [SWTT, List[SWTTScorerInput]]]
from arg.counter_arg_retrieval.build_dataset.passage_scoring.split_passages import split_passages
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from bert_api.swtt.window_enum_policy import get_enum_policy_30_to_400_50per_doc
from cache import load_from_pickle, save_to_pickle
from cpath import output_path
from trec.trec_parse import load_ranked_list_grouped


def load_ca3_swtt_passage():
    return load_from_pickle("ca_run3_swtt_passages")


def main():
    enum_policy = get_enum_policy_30_to_400_50per_doc()
    docs: List[Tuple[str, SegmentwiseTokenizedText]] = load_from_pickle("ca_run3_swtt")
    rlg_path = os.path.join(output_path, "ca_building", "run3", "q_res_2.filtered.txt")
    rlg = load_ranked_list_grouped(rlg_path)
    doc_as_passage_dict = split_passages(docs, rlg, enum_policy)
    save_to_pickle(doc_as_passage_dict, "ca_run3_swtt_passages")


if __name__ == "__main__":
    main()
