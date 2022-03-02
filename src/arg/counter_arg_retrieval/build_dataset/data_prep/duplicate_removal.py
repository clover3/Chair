from typing import List, Tuple

from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from list_lib import right
from trec.ranked_list_util import remove_duplicates_from_ranked_list


def remove_duplicates(ranked_list_grouped, docs: List[Tuple[str, SegmentwiseTokenizedText]]):
    n_docs = len(docs)
    duplicate_indices = SegmentwiseTokenizedText.get_duplicate(right(docs))
    print("duplicate_indices {} ".format(len(duplicate_indices)))
    duplicate_doc_ids = [docs[idx][0] for idx in duplicate_indices]
    docs = [e for idx, e in enumerate(docs) if idx not in duplicate_indices]
    print("{} docs after filtering (from {})".format(len(docs), n_docs))
    new_ranked_list_grouped = remove_duplicates_from_ranked_list(ranked_list_grouped, duplicate_doc_ids)
    return docs, duplicate_doc_ids, new_ranked_list_grouped
