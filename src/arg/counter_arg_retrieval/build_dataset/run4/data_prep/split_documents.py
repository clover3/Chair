import os
import pickle
from typing import Dict
from typing import List, Tuple

# Input: Ranked List, Enum Policy, SWTT
# Output:   Dict[doc_id, [SWTT, List[SWTTScorerInput]]]
from arg.counter_arg_retrieval.build_dataset.passage_scoring.split_passages import split_passages, PassageRange
from arg.counter_arg_retrieval.build_dataset.run4.run4_util import run4_rlg_filtered
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from bert_api.swtt.window_enum_policy import get_enum_policy_30_to_400_50per_doc
from cache import save_to_pickle
from cpath import output_path
from trec.trec_parse import load_ranked_list_grouped
from trec.types import DocID


def load_run4_swtt():
    max_job = 70
    all_docs = []
    for i in range(max_job):
        file_path = os.path.join(output_path,
                                 "ca_building", "run4", "pc_res_doc_jsonl_parsing",
                                 str(i)
                                 )
        docs: List[Tuple[str, SegmentwiseTokenizedText]] = pickle.load(open(file_path, "rb"))
        all_docs.extend(docs)
    return all_docs


def main():
    enum_policy = get_enum_policy_30_to_400_50per_doc()
    docs: List[Tuple[str, SegmentwiseTokenizedText]] = load_run4_swtt()
    print("Loaed {} docs".format(len(docs)))
    rlg_path = run4_rlg_filtered()
    rlg = load_ranked_list_grouped(rlg_path)

    rlg_size = sum(map(len, rlg.values()))
    print(f"{rlg_size} ranked list entries")
    doc_as_passage_dict: Dict[DocID, Tuple[SegmentwiseTokenizedText, List[PassageRange]]] = \
        split_passages(docs, rlg, enum_policy)
    save_to_pickle(doc_as_passage_dict, "ca_run4_swtt_passages")


if __name__ == "__main__":
    main()
