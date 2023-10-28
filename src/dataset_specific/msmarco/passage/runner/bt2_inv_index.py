from typing import List, Iterable, Tuple

from adhoc.build_index import build_inverted_index_plus, save_inv_index_to_pickle
from cpath import output_path
from dataset_specific.msmarco.passage.doc_indexing.index_path_helper import \
    get_bm25_bt2_resource_path_helper
from misc_lib import TELI
from misc_lib import path_join
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log


def merge_subwords(doc_tokens_str):
    doc_sbword_tokens = doc_tokens_str.split()
    merged_tokens = []
    current_token = ""

    for token in doc_sbword_tokens:
        if token.startswith("##"):
            current_token += token[2:]
        else:
            if current_token:  # If the current_token is not empty, append it to the merged_tokens list
                merged_tokens.append(current_token)
            current_token = token  # Start a new token

    if current_token:  # Add the last token if it's not empty
        merged_tokens.append(current_token)

    return merged_tokens


def iter_bert_tokenized_merged():
    save_path = path_join(output_path, "msmarco", "passage_bert_tokenized.tsv")
    f = open(save_path, "r")
    for line in f:
        doc_id, doc_tokens_str = line.split("\t")
        out_tokens = merge_subwords(doc_tokens_str)

        yield doc_id, out_tokens


def common_index_preprocessing():
    conf = get_bm25_bt2_resource_path_helper()
    collection_size = 8841823
    itr = iter_bert_tokenized_merged()
    itr = TELI(itr, collection_size)
    corpus_tokenized: Iterable[Tuple[str, List[str]]] = itr
    ignore_voca = set()
    c_log.info("Building inverted index")
    outputs = build_inverted_index_plus(
        corpus_tokenized,
        ignore_voca,
        num_docs=collection_size
    )
    c_log.info("Building inverted Done")
    save_inv_index_to_pickle(conf, outputs)


def main():
    with JobContext("inv_index_building_bt2"):
        common_index_preprocessing()


if __name__ == "__main__":
    main()