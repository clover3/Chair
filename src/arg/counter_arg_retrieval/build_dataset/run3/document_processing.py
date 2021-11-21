import os
from typing import List, Tuple

from cache import save_to_pickle, load_from_pickle
from cpath import output_path
from data_generator.tokenize_helper import TokenizedText
from data_generator.tokenizer_wo_tf import get_tokenizer
from galagos.doc_processor import jsonl_to_tokenized_text
from misc_lib import ceil_divide, Averager


def main():
    file_path = os.path.join(output_path, "ca_building", "run3", "docs.jsonl")
    print("Reading documents")
    f = open(file_path, "r")
    print("Read done")
    # iter = file_iterator_interval(f, 0, 100)
    iter = f
    output = jsonl_to_tokenized_text(iter, get_tokenizer(), 20000)
    save_to_pickle(output, "ca_run3_document_processed")


def size_check():
    docs: List[Tuple[str, TokenizedText]] = load_from_pickle("ca_run3_document_processed")

    window_size = 400
    skip = 200
    averager = Averager()
    for doc_id, doc in docs:
        n_sb = len(doc.sbword_tokens)
        n_window = ceil_divide((n_sb - (window_size-skip)), window_size)
        averager.append(n_window)

    print("Total {} runs, avg {}".format(sum(averager.history), averager.get_average()))


if __name__ == "__main__":
    size_check()
